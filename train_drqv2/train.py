# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')

from pathlib import Path
import csv
import time

import hydra
import numpy as np
import torch
# from dm_env import specs
# import dmc

import lumi_env
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(42)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = lumi_env.make(self.cfg.env)
        self.eval_env = lumi_env.make(self.cfg.env)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      lumi_env.ArraySpec((1,), np.float32, 'reward'),
                      lumi_env.ArraySpec((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def _write_cubelift_log(self, episode, frame, step, sample_count, avg_height, max_height):
        log_path = self.work_dir / 'cubelift.log'
        fieldnames = [
            'episode',
            'frame',
            'step',
            'sample_count',
            'cube_avg_height_above_table',
            'cube_max_height_above_table',
        ]
        rows = []
        if log_path.exists():
            with log_path.open('r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        old_episode = int(float(row.get('episode', -1)))
                    except (TypeError, ValueError):
                        old_episode = -1
                    if old_episode < episode:
                        rows.append(row)

        row = {
            'episode': int(episode),
            'frame': int(frame),
            'step': int(step),
            'sample_count': int(sample_count),
            'cube_avg_height_above_table': float(avg_height),
            'cube_max_height_above_table': float(max_height),
        }
        with log_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(row)

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        cube_height_stats = dict(sum=0.0, max=-np.inf, count=0)

        def reset_cube_height_stats():
            cube_height_stats['sum'] = 0.0
            cube_height_stats['max'] = -np.inf
            cube_height_stats['count'] = 0

        def record_cube_height():
            height = self.train_env.cube_height_above_table()
            cube_height_stats['sum'] += height
            cube_height_stats['max'] = max(cube_height_stats['max'], height)
            cube_height_stats['count'] += 1

        record_cube_height()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        profile_timings = bool(self.cfg.get('profile_timings', False))
        timing_sums = dict(act=0.0, update=0.0, env_step=0.0, replay_add=0.0, video=0.0)
        timing_counts = dict(act=0, update=0, env_step=0, replay_add=0, video=0)

        def add_timing(name, duration):
            if profile_timings:
                timing_sums[name] += duration
                timing_counts[name] += 1

        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                elapsed_time, total_time = self.timer.reset()
                episode_frame = episode_step * self.cfg.action_repeat
                height_count = cube_height_stats['count']
                if height_count > 0:
                    avg_cube_height = cube_height_stats['sum'] / height_count
                    max_cube_height = cube_height_stats['max']
                    self._write_cubelift_log(
                        self.global_episode,
                        self.global_frame,
                        self.global_step,
                        height_count,
                        avg_cube_height,
                        max_cube_height,
                    )

                with self.logger.log_and_dump_ctx(self.global_frame,
                                                  ty='train') as log:
                    log('fps', episode_frame / elapsed_time)
                    log('total_time', total_time)
                    log('episode_reward', episode_reward)
                    log('episode_length', episode_frame)
                    log('episode', self.global_episode)
                    log('buffer_size', len(self.replay_storage))
                    log('step', self.global_step)
                    if profile_timings:
                        for name in sorted(timing_sums):
                            count = max(1, timing_counts[name])
                            log(f'time_{name}_ms', 1000.0 * timing_sums[name] / count)
                        for name in timing_sums:
                            timing_sums[name] = 0.0
                            timing_counts[name] = 0

                # reset env
                time_step = self.train_env.reset()
                reset_cube_height_stats()
                record_cube_height()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                # if self.cfg.save_snapshot:
                #     self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                if self.cfg.save_snapshot:
                    self.save_eval_checkpoint()

            # sample action
            t0 = time.perf_counter()
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)
            add_timing('act', time.perf_counter() - t0)

            # try to update the agent. Episodes are only written to the
            # replay loader after time_step.last(), so updates must wait until
            # every replay worker can see at least one complete episode.
            min_episodes = max(1, self.cfg.replay_buffer_num_workers)
            replay_ready = (
                len(self.replay_storage) >= self.cfg.nstep
                and self.replay_storage.num_episodes >= min_episodes
            )
            if replay_ready and not seed_until_step(self.global_step):
                t0 = time.perf_counter()
                metrics = self.agent.update(self.replay_iter, self.global_step)
                add_timing('update', time.perf_counter() - t0)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            t0 = time.perf_counter()
            time_step = self.train_env.step(action)
            add_timing('env_step', time.perf_counter() - t0)
            record_cube_height()
            episode_reward += time_step.reward
            t0 = time.perf_counter()
            self.replay_storage.add(time_step)
            add_timing('replay_add', time.perf_counter() - t0)
            t0 = time.perf_counter()
            self.train_video_recorder.record(time_step.observation)
            add_timing('video', time.perf_counter() - t0)
            episode_step += 1
            self._global_step += 1

        if episode_step > 0:
            self.train_video_recorder.save(f'partial_{self.global_frame}.mp4')

    def save_snapshot(self, snapshot=None):
        if snapshot is None:
            snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_eval_checkpoint(self):
        checkpoint_dir = self.work_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint = checkpoint_dir / f'snapshot_{self.global_frame}.pt'
        self.save_snapshot(checkpoint)
        self.save_snapshot()

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config', version_base='1.3')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
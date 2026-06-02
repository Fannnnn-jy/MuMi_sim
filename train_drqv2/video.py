# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# try:
#     import cv2
# except ModuleNotFoundError:
#     cv2 = None
try:
    import imageio
except ModuleNotFoundError:
    imageio = None
import numpy as np


# def _resize_frame(frame, size):
#     if cv2 is None or frame.shape[:2] == (size, size):
#         return frame
#     return cv2.resize(frame, dsize=(size, size), interpolation=cv2.INTER_CUBIC)


def _save_video(path, frames, fps):
    if not frames:
        return
    if imageio is not None:
        imageio.mimsave(str(path), frames, fps=fps)
        return
    raise RuntimeError('Video saving requested, but imageio is not installed.')
    # if cv2 is None:
    #     raise RuntimeError('Video saving requested, but neither imageio nor cv2 is installed.')
    # first = np.asarray(frames[0], dtype=np.uint8)
    # height, width = first.shape[:2]
    # writer = cv2.VideoWriter(
    #     str(path),
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     float(fps),
    #     (width, height),
    # )
    # if not writer.isOpened():
    #     raise RuntimeError(f'Failed to open video writer for {path}')
    # try:
    #     for frame in frames:
    #         frame = np.asarray(frame, dtype=np.uint8)
    #         if frame.shape[:2] != (height, width):
    #             frame = _resize_frame(frame, width)
    #         if frame.ndim == 2:
    #             frame = np.repeat(frame[:, :, None], 3, axis=2)
    #         writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    # finally:
    #     writer.release()


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            _save_video(path, self.frames, self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = obs[:3].transpose(1, 2, 0)
            # frame = _resize_frame(frame, self.render_size)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            _save_video(path, self.frames, self.fps)

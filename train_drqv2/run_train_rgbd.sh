#!/bin/bash
 
while true; do
    uv run python train.py hydra.run.dir=./exp_local/2026.06.01/rgbd env.observation_mode=rgbd
done > train.log 2>&1 &

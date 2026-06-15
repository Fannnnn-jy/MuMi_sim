#!/bin/bash
 
while true; do
    sleep 10
    uv run python train.py hydra.run.dir=./exp_local/count_height/rgbd env.observation_mode=rgbd
done > train.log 2>&1 &

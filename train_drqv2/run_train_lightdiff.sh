#!/bin/bash
 
while true; do
    sleep 10
    uv run python train.py hydra.run.dir=./exp_local/count_height/light_diff env.observation_mode=light_diff
done > train.log 2>&1 &

#!/bin/bash

n=0
echo "Starting training runs..."
while [ ${n} -lt 10 ]; do
    echo "Starting run ${n}"
    uv run python train.py hydra.run.dir=./exp_local/2026.06.01/rgb env.observation_mode=rgb_stack
    # uv run python train.py hydra.run.dir=./exp_local/2026.06.01/light_diff env.observation_mode=light_diff
    # uv run python train.py hydra.run.dir=./exp_local/2026.06.01/rgbd env.observation_mode=rgbd
    echo "Finished run ${n}"
    n=$((n + 1))
done

 

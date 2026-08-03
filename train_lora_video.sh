#!/bin/bash

set -e 


python -m playground.train_lora_video_percept \
    --cfg configs/lora_video_train_cfg.yaml \
    --gpu_ids 0 \
    --stage 5 \
    --save_dir "./checkpoints/lora_video_percept_qp0" \
    --epochs 50  

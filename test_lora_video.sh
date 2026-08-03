#!/bin/bash

set -e

python -m playground.test_lora_video_model \
    --root_dir D:/video-communication-dataset/video/UVG/sequences \
    --dataset UVG \
    --sep_len 96 --gop_size -1 \
    --save_dir ./results \
    --device cuda \
    --qps 0 8 \
    --intra_model "F:/research-code/2026/pretrained/aligned_cvpr2025_image.pth.tar" \
    --intra_lora_model "ckpts/latest_lora_checkpoint_qp8.pth.tar" \
    --inter_model "F:/research-code/2026/pretrained/aligned_cvpr2025_video.pth.tar" \
    --inter_lora_model "ckpts/latest_lora_video_checkpoint_qp8.pth.tar"


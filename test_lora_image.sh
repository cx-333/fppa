#!/bin/bash

set -e 

python -m playground.test_lora_image_model \
    --data_path "D:/video-communication-dataset/image/kodak" \
    --quant_level 0 16 32 42 56 63 \
    --cuda \
    --model_path "F:/research-code/2026/pretrained/aligned_cvpr2025_image.pth.tar" \
    --lora_path "ckpts/latest_lora_checkpoint_qp63.pth.tar"

# "F:/research-supplement-code/视频压缩/DCVC/checkpoints/latest_model.pth.tar"

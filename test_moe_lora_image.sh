#!/bin/bash

set -e 

# mode: fidelity or perception 

python -m playground.test_moe_lora_image_model \
    --data_path "./assets" \
    --quant_level 15 \
    --cuda \
    --mode fidelity \
    --save_dir ./results/fidelity \
    --save \
    --model_path "ckpts/pretrained_mse_model.pth.tar" \
    --lora_path "ckpts/latest_moe_lora_checkpoint_qp0_63_qcond.pth.tar"


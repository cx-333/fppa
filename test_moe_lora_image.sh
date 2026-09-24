#!/bin/bash

set -e 

# mode: fidelity or perception 
# quant_level: single value or multiple values (0~63)

python -m playground.test_moe_lora_image_model \
    --data_path "./assets" \
    --quant_level 15 \
    --cuda \
    --mode fidelity \
    --save_dir ./results/fidelity \
    --save \
    --model_path "ckpts/pretrained_mse_model.pth.tar" \
    --lora_path "ckpts/latest_moe_lora_checkpoint_qp0_63_qcond.pth.tar"


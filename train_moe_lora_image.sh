#!/bin/bash

set -e 


python -m playground.train_moe_lora_image_percept \
    --cfg configs/moe_lora_image_train_cfg.yaml
#!/bin/bash

set -e 


python -m playground.train_lora_image_percept \
    --cfg configs/lora_image_train_cfg.yaml


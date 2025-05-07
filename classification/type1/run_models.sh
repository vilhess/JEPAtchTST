#!/bin/bash

sizes=("small" "base" "big") # 

for SIZE in "${sizes[@]}"; do
    echo "Running experiments on size: $SIZE"
    python main.py encoder=config_encoder_$SIZE
    done
done
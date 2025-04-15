#!/bin/bash

# Define datasets to iterate over
datasets=("etth1" "etth2" "ettm1" "ettm2")

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py forecasting=$DATASET
done

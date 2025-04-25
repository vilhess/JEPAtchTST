#!/bin/bash

# Define datasets to iterate over
datasets=( "weather" "national_illness") # electricity "etth1" "etth2" "ettm1" "ettm2"  "exchange_rate"

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=False
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=True
done

#!/bin/bash

# Define datasets to iterate over
datasets=(  "etth2" "ettm1" "ettm2"  "exchange_rate" "weather" "national_illness") # electricity "etth1"

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=False
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=True
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=False forecasting.revin=True
    python main.py forecasting=$DATASET encoder=config_encoder_big forecasting.univariate=True forecasting.revin=True
done

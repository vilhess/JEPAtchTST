#!/bin/bash

# Define datasets to iterate over
sizes=( "small" "base" "big" )
datasets=("etth1" "etth2" "ettm1" "ettm2" "exchange_rate" "weather" "national_illness" ) # electricity 

# Loop through each size and dataset
for SIZE in "${sizes[@]}"; do
    echo "Running experiments on size: $SIZE"
    for DATASET in "${datasets[@]}"; do
        echo "Running experiments on dataset: $DATASET"
        python main.py forecasting=$DATASET encoder=config_encoder_$SIZE forecasting.univariate=False
        python main.py forecasting=$DATASET encoder=config_encoder_$SIZE forecasting.univariate=True
        python main.py forecasting=$DATASET encoder=config_encoder_$SIZE forecasting.univariate=False forecasting.revin=True
        python main.py forecasting=$DATASET encoder=config_encoder_$SIZE forecasting.univariate=True forecasting.revin=True
    done
done

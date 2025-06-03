#!/bin/bash

# Define datasets to iterate over
datasets=("etth1" "etth2" "ettm1" "ettm2" "exchange_rate" "weather" "national_illness" )

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py forecasting=$DATASET forecasting.univariate=False forecasting.revin=True
    python main.py forecasting=$DATASET forecasting.univariate=True forecasting.revin=True
done
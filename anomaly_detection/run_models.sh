#!/bin/bash

# Define datasets to iterate over
datasets=("msl" "swat" "smd" "smap" ) # "nyc_taxi" "ec2_request_latency_system_failure" 

for DATASET in "${datasets[@]}"; do
    echo "Running experiments on dataset: $DATASET"
    python main.py anomaly_detection=$DATASET
done

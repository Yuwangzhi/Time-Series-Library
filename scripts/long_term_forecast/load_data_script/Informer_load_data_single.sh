#!/bin/bash

# Informer model for load_data dataset
# Using past 96 time points to predict future time points (192)

# export CUDA_VISIBLE_DEVICES=0

model_name=Informer

echo "========================================"
echo "Starting Informer experiments for load_data dataset"
echo "Running 4 experiments: 96->192"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 96->192 prediction
echo ""
echo "Starting Experiment 2/4: 96->192 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_192_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_192 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1
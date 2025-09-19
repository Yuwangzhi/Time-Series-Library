#!/bin/bash

# PatchTST model for load_data dataset
# Using past 96 time points to predict future time points

# export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

echo "Starting PatchTST training for load_data predictions..."
echo "Running two experiments: 96->96 and 96->192"

# 96->96 prediction
echo "========================================"
echo "Starting 96->96 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_96_PatchTST_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_96_PatchTST \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --n_heads 2 \
  --patch_len 16 \
  --train_epochs 1

echo "96->96 prediction completed!"

# 96->192 prediction
echo "========================================"
echo "Starting 96->192 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_192_PatchTST_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_192_PatchTST \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --n_heads 2 \
  --patch_len 16 \
  --train_epochs 1

echo "96->192 prediction completed!"
echo "========================================"
echo "All training completed! Results saved in:"
echo "- Model weights: ./checkpoints/long_term_forecast_load_data_96_*_PatchTST_*/"
echo "- Test results and logs: ./results/long_term_forecast_load_data_96_*_PatchTST_*/"
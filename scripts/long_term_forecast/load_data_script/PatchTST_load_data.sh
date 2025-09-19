#!/bin/bash

# PatchTST model for load_data dataset
# Using past 96 time points to predict future time points (96, 192, 336, 720)

export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

echo "========================================"
echo "Starting PatchTST experiments for load_data dataset"
echo "Running 4 experiments: 96->96, 96->192, 96->336, 96->720"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 96->96 prediction
echo ""
echo "Starting Experiment 1/4: 96->96 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_96_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_96 \
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
  --n_heads 4

echo "Experiment 1/4 completed!"

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
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --n_heads 4

echo "Experiment 2/4 completed!"

# 96->336 prediction
echo ""
echo "Starting Experiment 3/4: 96->336 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_336_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_336 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --n_heads 4

echo "Experiment 3/4 completed!"

# 96->720 prediction
echo ""
echo "Starting Experiment 4/4: 96->720 prediction..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_96_720_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path lf_load_data_20210101-20250807_processed.csv \
  --model_id load_data_96_720 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --n_heads 4

echo "Experiment 4/4 completed!"

echo ""
echo "========================================"
echo "All PatchTST experiments completed!"
echo "Results and logs available in:"
echo "- ./results/long_term_forecast_load_data_96_96_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_96_192_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_96_336_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_96_720_*/experiment_log.txt"
echo "========================================"
#!/bin/bash

# Informer model for load_data dataset - 60 minute granularity
# Using past 96 time points (96 hours / 4 days) to predict future time points (96, 192, 336, 720)

# export CUDA_VISIBLE_DEVICES=0

model_name=Informer

echo "========================================"
echo "Starting Informer experiments for load_data dataset (60-minute granularity)"
echo "Running 4 experiments: 336->96, 336->192, 336->336, 336->720"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 336->96 prediction (14 days -> 4 days)
echo ""
echo "Starting Experiment 1/4: 336->96 prediction (14d->4d)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_336_96_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_336_96 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 1/4 completed!"

# 336->192 prediction (14 days -> 8 days)
echo ""
echo "Starting Experiment 2/4: 336->192 prediction (14d->8d)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_336_192_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_336_192 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 336 \
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

echo "Experiment 2/4 completed!"

# 336->336 prediction (14 days -> 14 days)
echo ""
echo "Starting Experiment 3/4: 336->336 prediction (14d->14d)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_336_336_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_336_336 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 3/4 completed!"

# 336->720 prediction (14 days -> 30 days)
echo ""
echo "Starting Experiment 4/4: 336->720 prediction (14d->30d)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_336_720_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_336_720 \
  --model $model_name \
  --data load_data \
  --features S \
  --seq_len 336 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 4/4 completed!"

echo ""
echo "========================================"
echo "All Informer experiments completed (60-minute granularity)!"
echo "Results and logs available in:"
echo "- ./results/long_term_forecast_load_data_60min_336_96_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_336_192_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_336_336_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_336_720_*/experiment_log.txt"
echo "========================================"
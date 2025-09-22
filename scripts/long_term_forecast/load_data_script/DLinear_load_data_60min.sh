#!/bin/bash

# DLinear model for load_data dataset (60-minute granularity)
# Using past 96 time points to predict future time points (96, 192, 336, 720)

# export CUDA_VISIBLE_DEVICES=0

model_name=DLinear

echo "========================================"
echo "Starting DLinear experiments for load_data dataset (60-minute granularity)"
echo "Running 4 experiments: 96->96, 96->192, 96->336, 96->720"
echo "Data granularity: 60 minutes (1 hour)"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 96->96 prediction (96 hours -> 96 hours at 60-minute intervals)
echo ""
echo "Starting Experiment 1/4: 96->96 prediction (96h->96h)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_96_96_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_96_96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 1/4 completed!"

# 96->192 prediction (96 hours -> 192 hours at 60-minute intervals)
echo ""
echo "Starting Experiment 2/4: 96->192 prediction (96h->192h)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_96_192_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_96_192 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 2/4 completed!"

# 96->336 prediction (96 hours -> 336 hours at 60-minute intervals)
echo ""
echo "Starting Experiment 3/4: 96->336 prediction (96h->336h)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_96_336_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_96_336 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 3/4 completed!"

# 96->720 prediction (96 hours -> 720 hours at 60-minute intervals)
echo ""
echo "Starting Experiment 4/4: 96->720 prediction (96h->720h)..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_60min_96_720_*/"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250807_60min.csv \
  --model_id load_data_60min_96_720 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --itr 1

echo "Experiment 4/4 completed!"

echo ""
echo "========================================"
echo "All DLinear experiments completed for 60-minute granularity data!"
echo "Time interpretations:"
echo "- 96 time steps = 96 hours (96 × 60 min = 4 days)"
echo "- 192 time steps = 192 hours (192 × 60 min = 8 days)"
echo "- 336 time steps = 336 hours (336 × 60 min = 14 days)"
echo "- 720 time steps = 720 hours (720 × 60 min = 30 days)"
echo ""
echo "Results and logs available in:"
echo "- ./results/long_term_forecast_load_data_60min_96_96_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_96_192_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_96_336_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_60min_96_720_*/experiment_log.txt"
echo "========================================"
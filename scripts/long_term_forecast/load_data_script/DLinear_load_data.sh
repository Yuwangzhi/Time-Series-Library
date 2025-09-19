#!/bin/bash

# DLinear model for load_data dataset
# Using past 96 time points to predict future 96 time points

model_name=DLinear

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
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --target value \
  --des 'Exp' \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --train_epochs 10 \
  --patience 3 \
  --itr 1
#!/bin/bash

# Multiple models for load_data dataset (15-minute granularity)
# Using past 96 time points to predict future 96 time points

# Common parameters
root_path="./dataset/load_data/hf_load_data/"
data_path="hf_load_data_20210101-20250807_15min.csv"
data_name="custom"
features="S"
seq_len=96
label_len=48
pred_len=96
enc_in=1
dec_in=1
c_out=1
target="value"
batch_size=32
learning_rate=0.0001
train_epochs=10
patience=3

echo "========================================================================"
echo "              Multiple Models Load Data Experiments (15-min)"
echo "========================================================================"
echo "Data granularity: 15 minutes"
echo "Time interpretation: 96 time steps = 24 hours (96 × 15 min)"
echo "Prediction: 24 hours history → 24 hours future"
echo "========================================================================"

echo ""
echo "========== Running DLinear (15-min) =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_15min_96_96_DLinear \
  --model DLinear \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --target $target \
  --des 'Exp' \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --itr 1

echo ""
echo "========== Running Transformer (15-min) =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_15min_96_96_Transformer \
  --model Transformer \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --target $target \
  --des 'Exp' \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --itr 1

echo ""
echo "========== Running TimesNet (15-min) =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_15min_96_96_TimesNet \
  --model TimesNet \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --target $target \
  --des 'Exp' \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --itr 1

echo ""
echo "========== Running PatchTST (15-min) =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_15min_96_96_PatchTST \
  --model PatchTST \
  --data $data_name \
  --features $features \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --n_heads 4 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --target $target \
  --des 'Exp' \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --itr 1

echo ""
echo "========================================================================"
echo "All Multiple Models Experiments Completed (15-minute granularity)!"
echo "Time interpretation: 96 time steps = 24 hours (96 × 15 min)"
echo "Results available in ./results/long_term_forecast_load_data_15min_96_96_*/"
echo "========================================================================"
#!/bin/bash

# Multiple models for load_data dataset
# Using past 96 time points to predict future 96 time points

# Common parameters
root_path="./dataset/load_data/hf_load_data/"
data_path="lf_load_data_20210101-20250807_processed.csv"
data_name="load_data"
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

echo "========== Running DLinear =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_96_96_DLinear \
  --model DLinear \
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

echo "========== Running Transformer =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_96_96_Transformer \
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

echo "========== Running TimesNet =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_96_96_TimesNet \
  --model TimesNet \
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
  --top_k 5 \
  --num_kernels 6 \
  --itr 1

echo "========== Running PatchTST =========="
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path \
  --data_path $data_path \
  --model_id load_data_96_96_PatchTST \
  --model PatchTST \
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
  --patch_len 16 \
  --stride 8 \
  --itr 1
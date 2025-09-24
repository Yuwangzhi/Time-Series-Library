# export CUDA_VISIBLE_DEVICES=1

model_name=TimeXer

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
  --e_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --batch_size 32 \
  --des 'exp' \
  --itr 1


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_192 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 192 \
#   --e_layers 2 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 128 \
#   --batch_size 4 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_336 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 1024 \
#   --batch_size 16 \
#   --itr 1

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1.csv \
#   --model_id ETTh1_96_720 \
#   --model $model_name \
#   --data ETTh1 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 1 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --d_model 256 \
#   --d_ff 1024 \
#   --batch_size 16 \
#   --itr 1

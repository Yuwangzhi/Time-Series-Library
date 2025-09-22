#!/usr/bin/env python3
"""
简化版批量生成15分钟和60分钟粒度的模型训练脚本
"""

import os
import re
from pathlib import Path

def get_model_specific_params(model_name):
    """根据模型名称返回特定参数"""
    if model_name in ['Informer']:
        return """  --e_layers 3 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name in ['Transformer']:
        return """  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name == 'Autoformer':
        return """  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --moving_avg 25 \\"""
    elif model_name == 'FEDformer':
        return """  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --base 'legendre' \\
  --cross_activation 'tanh' \\"""
    elif model_name == 'TimesNet':
        return """  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\"""
    elif model_name == 'PatchTST':
        return """  --e_layers 3 \\
  --n_heads 4 \\
  --d_model 128 \\
  --d_ff 256 \\
  --dropout 0.2 \\
  --fc_dropout 0.2 \\
  --head_dropout 0 \\
  --patch_len 16 \\
  --stride 8 \\"""
    elif model_name == 'iTransformer':
        return """  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 512 \\
  --d_ff 512 \\"""
    elif model_name == 'Crossformer':
        return """  --e_layers 3 \\
  --d_layers 1 \\
  --d_model 256 \\
  --d_ff 512 \\
  --n_heads 4 \\
  --seg_len 6 \\
  --win_size 2 \\
  --factor 10 \\"""
    else:
        return ""

def create_script_template(model_name, granularity):
    """创建脚本模板"""
    if granularity == '15min':
        data_file = 'hf_load_data_20210101-20250807_15min.csv'
        description = '15-minute granularity'
        time_desc = '15 minutes'
        exp_descs = ['24h->24h', '24h->48h', '24h->84h', '24h->180h']
        time_interps = [
            '- 96 time steps = 24 hours (96 × 15 min)',
            '- 192 time steps = 48 hours (192 × 15 min)',
            '- 336 time steps = 84 hours (336 × 15 min)',
            '- 720 time steps = 180 hours (720 × 15 min)'
        ]
    else:  # 60min
        data_file = 'hf_load_data_20210101-20250807_60min.csv'
        description = '60-minute granularity'
        time_desc = '60 minutes (1 hour)'
        exp_descs = ['96h->96h', '96h->192h', '96h->336h', '96h->720h']
        time_interps = [
            '- 96 time steps = 96 hours (96 × 60 min = 4 days)',
            '- 192 time steps = 192 hours (192 × 60 min = 8 days)',
            '- 336 time steps = 336 hours (336 × 60 min = 14 days)',
            '- 720 time steps = 720 hours (720 × 60 min = 30 days)'
        ]
    
    model_params = get_model_specific_params(model_name)
    
    script_content = f"""#!/bin/bash

# {model_name} model for load_data dataset ({description})
# Using past 96 time points to predict future time points (96, 192, 336, 720)

# export CUDA_VISIBLE_DEVICES=0

model_name={model_name}

echo "========================================"
echo "Starting {model_name} experiments for load_data dataset ({description})"
echo "Running 4 experiments: 96->96, 96->192, 96->336, 96->720"
echo "Data granularity: {time_desc}"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 96->96 prediction
echo ""
echo "Starting Experiment 1/4: 96->96 prediction ({exp_descs[0]})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{granularity}_96_96_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {data_file} \\
  --model_id load_data_{granularity}_96_96 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 96 \\{model_params}
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 1/4 completed!"

# 96->192 prediction
echo ""
echo "Starting Experiment 2/4: 96->192 prediction ({exp_descs[1]})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{granularity}_96_192_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {data_file} \\
  --model_id load_data_{granularity}_96_192 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 192 \\{model_params}
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 2/4 completed!"

# 96->336 prediction
echo ""
echo "Starting Experiment 3/4: 96->336 prediction ({exp_descs[2]})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{granularity}_96_336_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {data_file} \\
  --model_id load_data_{granularity}_96_336 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 336 \\{model_params}
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 3/4 completed!"

# 96->720 prediction
echo ""
echo "Starting Experiment 4/4: 96->720 prediction ({exp_descs[3]})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{granularity}_96_720_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {data_file} \\
  --model_id load_data_{granularity}_96_720 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 720 \\{model_params}
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 4/4 completed!"

echo ""
echo "========================================"
echo "All {model_name} experiments completed for {description} data!"
echo "Time interpretations:"
"""
    
    for interp in time_interps:
        script_content += f'echo "{interp}"\n'
    
    script_content += f"""echo ""
echo "Results and logs available in:"
echo "- ./results/long_term_forecast_load_data_{granularity}_96_96_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity}_96_192_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity}_96_336_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity}_96_720_*/experiment_log.txt"
echo "========================================"
"""
    
    return script_content

def main():
    """主函数"""
    script_dir = Path(r"d:\ywz_experiment_papers\ywz_7_electric\Time-Series-Library\scripts\long_term_forecast\load_data_script")
    
    # 需要处理的模型列表
    models = [
        'Autoformer',
        'Crossformer', 
        'FEDformer',
        'iTransformer',
        'MICN',
        'Nonstationary_Transformer',
        'PatchTST',
        'SegRNN',
        'TimeMixer',
        'TimesNet',
        'Transformer',
        'TSMixer'
    ]
    
    print("开始批量创建15分钟和60分钟粒度的训练脚本...")
    print("="*60)
    
    created_files = []
    
    for model_name in models:
        print(f"\n处理模型: {model_name}")
        
        # 创建15分钟版本
        script_15min = create_script_template(model_name, '15min')
        file_15min = script_dir / f"{model_name}_load_data_15min.sh"
        
        with open(file_15min, 'w', encoding='utf-8') as f:
            f.write(script_15min)
        print(f"✅ 创建成功: {file_15min.name}")
        created_files.append(str(file_15min))
        
        # 创建60分钟版本
        script_60min = create_script_template(model_name, '60min')
        file_60min = script_dir / f"{model_name}_load_data_60min.sh"
        
        with open(file_60min, 'w', encoding='utf-8') as f:
            f.write(script_60min)
        print(f"✅ 创建成功: {file_60min.name}")
        created_files.append(str(file_60min))
    
    print("\n" + "="*60)
    print(f"批量创建完成！共创建 {len(created_files)} 个文件")

if __name__ == "__main__":
    main()
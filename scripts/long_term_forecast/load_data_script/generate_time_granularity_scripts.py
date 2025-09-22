#!/usr/bin/env python3
"""
批量生成15分钟和60分钟粒度的模型训练脚本
为所有load_data_script目录下的模型脚本创建对应的15min和60min版本
"""

import os
import re
from pathlib import Path

def create_time_granularity_script(original_file, granularity):
    """
    基于原始脚本创建指定时间粒度的脚本
    
    Args:
        original_file: 原始脚本文件路径
        granularity: 时间粒度 ('15min' 或 '60min')
    """
    if not os.path.exists(original_file):
        print(f"警告: 原始文件不存在: {original_file}")
        return None
    
    # 读取原始脚本内容
    with open(original_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取模型名称
    model_match = re.search(r'model_name=(\w+)', content)
    if not model_match:
        print(f"警告: 无法从 {original_file} 中提取模型名称")
        return None
    
    model_name = model_match.group(1)
    
    # 设置时间粒度相关信息
    if granularity == '15min':
        granularity_info = {
            'suffix': '15min',
            'file_suffix': '_15min',
            'data_file': 'hf_load_data_20210101-20250807_15min.csv',
            'description': '15-minute granularity',
            'time_desc': '15 minutes',
            'time_interpretations': [
                "- 96 time steps = 24 hours (96 × 15 min)",
                "- 192 time steps = 48 hours (192 × 15 min)",
                "- 336 time steps = 84 hours (336 × 15 min)",
                "- 720 time steps = 180 hours (720 × 15 min)"
            ],
            'experiment_descriptions': [
                "24h->24h", "24h->48h", "24h->84h", "24h->180h"
            ]
        }
    else:  # 60min
        granularity_info = {
            'suffix': '60min',
            'file_suffix': '_60min',
            'data_file': 'hf_load_data_20210101-20250807_60min.csv',
            'description': '60-minute granularity',
            'time_desc': '60 minutes (1 hour)',
            'time_interpretations': [
                "- 96 time steps = 96 hours (96 × 60 min = 4 days)",
                "- 192 time steps = 192 hours (192 × 60 min = 8 days)",
                "- 336 time steps = 336 hours (336 × 60 min = 14 days)",
                "- 720 time steps = 720 hours (720 × 60 min = 30 days)"
            ],
            'experiment_descriptions': [
                "96h->96h", "96h->192h", "96h->336h", "96h->720h"
            ]
        }
    
    # 创建新脚本内容
    new_content = f"""#!/bin/bash

# {model_name} model for load_data dataset ({granularity_info['description']})
# Using past 96 time points to predict future time points (96, 192, 336, 720)

# export CUDA_VISIBLE_DEVICES=0

model_name={model_name}

echo "========================================"
echo "Starting {model_name} experiments for load_data dataset ({granularity_info['description']})"
echo "Running 4 experiments: 96->96, 96->192, 96->336, 96->720"
echo "Data granularity: {granularity_info['time_desc']}"
echo "Each experiment will generate its own log file in results folder"
echo "========================================"

# 96->96 prediction
echo ""
echo "Starting Experiment 1/4: 96->96 prediction ({granularity_info['experiment_descriptions'][0]})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{granularity_info['suffix']}_96_96_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {granularity_info['data_file']} \\
  --model_id load_data_{granularity_info['suffix']}_96_96 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 96 \\"""
    
    # 添加模型特定的参数
    if model_name in ['Informer', 'Transformer']:
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name == 'Autoformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --moving_avg 25 \\"""
    elif model_name == 'FEDformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --base 'legendre' \\
  --cross_activation 'tanh' \\"""
    elif model_name in ['TimesNet']:
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\"""
    elif model_name in ['PatchTST']:
        new_content += """
  --e_layers 3 \\
  --n_heads 4 \\
  --d_model 128 \\
  --d_ff 256 \\
  --dropout 0.2 \\
  --fc_dropout 0.2 \\
  --head_dropout 0 \\
  --patch_len 16 \\
  --stride 8 \\"""
    elif model_name == 'iTransformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 512 \\
  --d_ff 512 \\"""
    elif model_name == 'Crossformer':
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --d_model 256 \\
  --d_ff 512 \\
  --n_heads 4 \\
  --seg_len 6 \\
  --win_size 2 \\
  --factor 10 \\"""
    
    # 添加通用参数
    new_content += """
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 1/4 completed!"

# 96->192 prediction
echo ""
echo "Starting Experiment 2/4: 96->192 prediction ({})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{}_96_192_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {} \\
  --model_id load_data_{}_96_192 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 192 \\""".format(
        granularity_info['experiment_descriptions'][1],
        granularity_info['suffix'],
        granularity_info['data_file'],
        granularity_info['suffix']
    )
    
    # 重复模型特定参数和通用参数（为了保持一致性）
    if model_name in ['Informer', 'Transformer']:
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name == 'Autoformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --moving_avg 25 \\"""
    elif model_name == 'FEDformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --base 'legendre' \\
  --cross_activation 'tanh' \\"""
    elif model_name in ['TimesNet']:
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\"""
    elif model_name in ['PatchTST']:
        new_content += """
  --e_layers 3 \\
  --n_heads 4 \\
  --d_model 128 \\
  --d_ff 256 \\
  --dropout 0.2 \\
  --fc_dropout 0.2 \\
  --head_dropout 0 \\
  --patch_len 16 \\
  --stride 8 \\"""
    elif model_name == 'iTransformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 512 \\
  --d_ff 512 \\"""
    elif model_name == 'Crossformer':
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --d_model 256 \\
  --d_ff 512 \\
  --n_heads 4 \\
  --seg_len 6 \\
  --win_size 2 \\
  --factor 10 \\"""
    
    new_content += """
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 2/4 completed!"

# 96->336 prediction
echo ""
echo "Starting Experiment 3/4: 96->336 prediction ({})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{}_96_336_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {} \\
  --model_id load_data_{}_96_336 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 336 \\""".format(
        granularity_info['experiment_descriptions'][2],
        granularity_info['suffix'],
        granularity_info['data_file'],
        granularity_info['suffix']
    )
    
    # 再次重复参数
    if model_name in ['Informer', 'Transformer']:
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name == 'Autoformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --moving_avg 25 \\"""
    elif model_name == 'FEDformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --base 'legendre' \\
  --cross_activation 'tanh' \\"""
    elif model_name in ['TimesNet']:
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\"""
    elif model_name in ['PatchTST']:
        new_content += """
  --e_layers 3 \\
  --n_heads 4 \\
  --d_model 128 \\
  --d_ff 256 \\
  --dropout 0.2 \\
  --fc_dropout 0.2 \\
  --head_dropout 0 \\
  --patch_len 16 \\
  --stride 8 \\"""
    elif model_name == 'iTransformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 512 \\
  --d_ff 512 \\"""
    elif model_name == 'Crossformer':
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --d_model 256 \\
  --d_ff 512 \\
  --n_heads 4 \\
  --seg_len 6 \\
  --win_size 2 \\
  --factor 10 \\"""
    
    new_content += """
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 3/4 completed!"

# 96->720 prediction
echo ""
echo "Starting Experiment 4/4: 96->720 prediction ({})..."
echo "Results will be saved in: ./results/long_term_forecast_load_data_{}_96_720_*/"

python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path ./dataset/load_data/hf_load_data/ \\
  --data_path {} \\
  --model_id load_data_{}_96_720 \\
  --model $model_name \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 720 \\""".format(
        granularity_info['experiment_descriptions'][3],
        granularity_info['suffix'],
        granularity_info['data_file'],
        granularity_info['suffix']
    )
    
    # 最后一次重复参数
    if model_name in ['Informer', 'Transformer']:
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --factor 3 \\"""
    elif model_name == 'Autoformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --moving_avg 25 \\"""
    elif model_name == 'FEDformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --factor 3 \\
  --base 'legendre' \\
  --cross_activation 'tanh' \\"""
    elif model_name in ['TimesNet']:
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\"""
    elif model_name in ['PatchTST']:
        new_content += """
  --e_layers 3 \\
  --n_heads 4 \\
  --d_model 128 \\
  --d_ff 256 \\
  --dropout 0.2 \\
  --fc_dropout 0.2 \\
  --head_dropout 0 \\
  --patch_len 16 \\
  --stride 8 \\"""
    elif model_name == 'iTransformer':
        new_content += """
  --e_layers 2 \\
  --d_layers 1 \\
  --d_model 512 \\
  --d_ff 512 \\"""
    elif model_name == 'Crossformer':
        new_content += """
  --e_layers 3 \\
  --d_layers 1 \\
  --d_model 256 \\
  --d_ff 512 \\
  --n_heads 4 \\
  --seg_len 6 \\
  --win_size 2 \\
  --factor 10 \\"""
    
    new_content += """
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --target value \\
  --des 'Exp' \\
  --itr 1

echo "Experiment 4/4 completed!"

echo ""
echo "========================================"
echo "All {model_name} experiments completed for {granularity_info['description']} data!"
echo "Time interpretations:"
{time_interp}
echo ""
echo "Results and logs available in:"
echo "- ./results/long_term_forecast_load_data_{granularity_info['suffix']}_96_96_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity_info['suffix']}_96_192_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity_info['suffix']}_96_336_*/experiment_log.txt"
echo "- ./results/long_term_forecast_load_data_{granularity_info['suffix']}_96_720_*/experiment_log.txt"
echo "========================================"
""".format(
        model_name=model_name,
        granularity_info=granularity_info,
        time_interp='\n'.join(f'echo "{interp}"' for interp in granularity_info['time_interpretations'])
    )
    
    # 生成输出文件名
    base_name = Path(original_file).stem
    output_file = str(Path(original_file).parent / f"{base_name}{granularity_info['file_suffix']}.sh")
    
    # 写入新脚本
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ 创建成功: {output_file}")
    return output_file

def main():
    """主函数"""
    script_dir = Path(r"d:\ywz_experiment_papers\ywz_7_electric\Time-Series-Library\scripts\long_term_forecast\load_data_script")
    
    # 需要处理的模型列表
    models = [
        'Autoformer_load_data.sh',
        'Crossformer_load_data.sh',
        'FEDformer_load_data.sh',
        'iTransformer_load_data.sh',
        'MICN_load_data.sh',
        'Nonstationary_Transformer_load_data.sh',
        'PatchTST_load_data.sh',
        'SegRNN_load_data.sh',
        'TimeMixer_load_data.sh',
        'TimesNet_load_data.sh',
        'Transformer_load_data.sh',
        'TSMixer_load_data.sh'
    ]
    
    print("开始批量创建15分钟和60分钟粒度的训练脚本...")
    print("="*60)
    
    created_files = []
    
    for model_script in models:
        original_path = script_dir / model_script
        
        if original_path.exists():
            print(f"\n处理模型: {model_script}")
            
            # 创建15分钟版本
            file_15min = create_time_granularity_script(str(original_path), '15min')
            if file_15min:
                created_files.append(file_15min)
            
            # 创建60分钟版本
            file_60min = create_time_granularity_script(str(original_path), '60min')
            if file_60min:
                created_files.append(file_60min)
        else:
            print(f"⚠️  警告: 文件不存在: {original_path}")
    
    print("\n" + "="*60)
    print(f"批量创建完成！共创建 {len(created_files)} 个文件")
    print("创建的文件列表:")
    for file_path in created_files:
        print(f"  ✅ {Path(file_path).name}")

if __name__ == "__main__":
    main()
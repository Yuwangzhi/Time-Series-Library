#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
将原始电力负荷数据转换为Time-Series-Library格式
用于长期预测任务

原始数据格式: index,date,value,
目标格式: date,value (符合Custom Dataset要求)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def preprocess_load_data(input_file, output_file=None):
    """
    预处理电力负荷数据
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径，如果为None则自动生成
    
    Returns:
        str: 输出文件路径
    """
    print(f"开始处理数据文件: {input_file}")
    
    # 读取原始数据
    try:
        df = pd.read_csv(input_file)
        print(f"原始数据形状: {df.shape}")
        print(f"原始列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None
    
    # 检查数据基本信息
    print("\n数据基本信息:")
    print(df.head())
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    
    # 数据预处理
    # 1. 去掉index列（如果存在）
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
        print("已删除index列")
    
    # 2. 去掉最后的空列（如果存在）
    df = df.dropna(axis=1, how='all')
    
    # 3. 确保只有date和value列
    # 删除所有空列
    df = df.dropna(axis=1, how='all')
    
    # 如果有多余的列，只保留前两列
    if len(df.columns) > 2:
        df = df.iloc[:, :2]  # 只保留前两列
        print(f"保留前两列，删除多余列")
    
    expected_columns = ['date', 'value']
    if list(df.columns) != expected_columns:
        print(f"警告: 列名不符合期望。期望: {expected_columns}, 实际: {list(df.columns)}")
        # 重命名列
        if len(df.columns) == 2:
            df.columns = expected_columns
            print(f"已重命名列为: {list(df.columns)}")
        else:
            print(f"错误: 列数不正确，期望2列，实际{len(df.columns)}列")
            return None
    
    # 4. 处理时间格式
    print("\n处理时间格式...")
    try:
        # 原始格式: 2021/1/1 0:00
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d %H:%M')
        # 转换为标准格式: 2021-01-01 00:00:00
        df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        print("时间格式转换成功")
    except Exception as e:
        print(f"时间格式转换失败: {e}")
        try:
            # 尝试自动解析
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            print("使用自动解析成功转换时间格式")
        except Exception as e2:
            print(f"时间格式转换失败: {e2}")
            return None
    
    # 5. 处理数值数据
    print("\n处理数值数据...")
    try:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        print(f"数值转换完成，缺失值数量: {df['value'].isnull().sum()}")
    except Exception as e:
        print(f"数值转换失败: {e}")
        return None
    
    # 6. 处理缺失值
    if df['value'].isnull().sum() > 0:
        print("检测到缺失值，使用前向填充...")
        df['value'] = df['value'].fillna(method='ffill')
        # 如果第一个值是NaN，使用后向填充
        df['value'] = df['value'].fillna(method='bfill')
        print(f"缺失值处理完成，剩余缺失值: {df['value'].isnull().sum()}")
    
    # 7. 去重和排序
    print("\n数据清理...")
    original_len = len(df)
    df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    print(f"去重后数据长度: {len(df)} (原始: {original_len})")
    
    # 8. 数据质量检查
    print("\n数据质量检查:")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"数值范围: {df['value'].min():.2f} 到 {df['value'].max():.2f}")
    print(f"数值均值: {df['value'].mean():.2f}")
    print(f"数值标准差: {df['value'].std():.2f}")
    
    # 检查时间间隔
    df_temp = df.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    time_diff = df_temp['date'].diff().dropna()
    most_common_interval = time_diff.mode()[0] if len(time_diff.mode()) > 0 else None
    print(f"最常见时间间隔: {most_common_interval}")
    
    # 9. 生成输出文件名
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        input_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(input_dir, f"{input_name}_processed.csv")
    
    # 10. 保存处理后的数据
    try:
        df.to_csv(output_file, index=False)
        print(f"\n数据预处理完成！")
        print(f"输出文件: {output_file}")
        print(f"最终数据形状: {df.shape}")
        print(f"最终列名: {list(df.columns)}")
        
        # 显示处理后数据的前几行
        print("\n处理后数据预览:")
        print(df.head(10))
        
        return output_file
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None

def generate_run_script(data_file, output_dir="./scripts/"):
    """
    生成运行脚本
    
    Args:
        data_file (str): 数据文件路径
        output_dir (str): 脚本输出目录
    """
    # 提取数据文件信息
    data_dir = os.path.dirname(data_file)
    data_name = os.path.basename(data_file)
    
    # 生成脚本内容
    script_content = f"""#!/bin/bash
# 电力负荷预测脚本
# 数据文件: {data_name}

# 基本配置
python -u run.py \\
  --task_name long_term_forecast \\
  --is_training 1 \\
  --root_path {data_dir}/ \\
  --data_path {data_name} \\
  --model_id load_data_96_96 \\
  --model TimesNet \\
  --data custom \\
  --features S \\
  --seq_len 96 \\
  --label_len 48 \\
  --pred_len 96 \\
  --enc_in 1 \\
  --dec_in 1 \\
  --c_out 1 \\
  --d_model 64 \\
  --d_ff 64 \\
  --top_k 5 \\
  --des 'Exp' \\
  --itr 1 \\
  --batch_size 32 \\
  --learning_rate 0.0001 \\
  --target value \\
  --freq t

# 预测96个时间步（8小时，每5分钟一个点）
echo "模型配置说明:"
echo "- 序列长度: 96 (8小时历史数据)"
echo "- 预测长度: 96 (8小时未来数据)"
echo "- 特征类型: S (单变量预测)"
echo "- 目标变量: value"
echo "- 频率: t (5分钟间隔)"
"""
    
    # 保存脚本
    os.makedirs(output_dir, exist_ok=True)
    script_file = os.path.join(output_dir, "run_load_forecast.sh")
    
    try:
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"\n运行脚本已生成: {script_file}")
        return script_file
    except Exception as e:
        print(f"生成脚本失败: {e}")
        return None

if __name__ == "__main__":
    # 数据文件路径
    input_file = r"D:\ywz_experiment_papers\ywz_7_electric\Time-Series-Library\dataset\load_data\hf_load_data\hf_load_data_20210101-20250807.csv"
    
    print("=" * 60)
    print("电力负荷数据预处理工具")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        exit(1)
    
    # 执行数据预处理
    output_file = preprocess_load_data(input_file)
    
    if output_file:
        print("\n" + "=" * 60)
        print("预处理成功完成！")
        
        # 生成运行脚本
        script_file = generate_run_script(output_file)
        
        print("\n使用说明:")
        print("1. 数据已转换为Time-Series-Library标准格式")
        print("2. 可直接用于长期预测任务")
        print("3. 建议的运行命令:")
        print(f"   python run.py --task_name long_term_forecast --data custom --root_path {os.path.dirname(output_file)}/ --data_path {os.path.basename(output_file)} --features S --target value")
        
    else:
        print("\n预处理失败！请检查输入数据格式。")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Load Analysis Tool
Generate CSV analysis for load_data model results
"""

import os
import numpy as np
import pandas as pd
import re
from pathlib import Path

def analyze_load_data_results():
    """
    Analyze load_data model results and save to CSV
    """
    
    print("🔍 Analyzing Load_Data Results...")
    
    results_dir = Path('./results')
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    # Collect results
    results = []
    
    for folder in results_dir.iterdir():
        if not folder.is_dir() or 'load_data' not in folder.name:
            continue
        
        # Parse folder name using regex
        try:
            pattern = r'long_term_forecast_load_data_(\d+)_(\d+)_([^_]+)_'
            match = re.search(pattern, folder.name)
            if match:
                seq_len = match.group(1)    # 96
                pred_len = match.group(2)   # 192, 336, etc.
                model = match.group(3)      # PatchTST, Informer, etc.
                input_output = f"{seq_len}->{pred_len}"
            else:
                continue
        except Exception as e:
            print(f"Error parsing {folder.name}: {e}")
            continue
        
        # Load metrics from .npy file
        metrics_file = folder / 'metrics.npy'
        metrics = None
        
        if metrics_file.exists():
            try:
                metrics_array = np.load(metrics_file)
                # Standard order: [MAE, MSE, RMSE, MAPE, MSPE]
                if len(metrics_array) >= 5:
                    metrics = {
                        'MAE': float(metrics_array[0]),
                        'MSE': float(metrics_array[1]),
                        'RMSE': float(metrics_array[2]),
                        'MAPE': float(metrics_array[3]),
                        'MSPE': float(metrics_array[4])
                    }
            except Exception as e:
                print(f"Error loading metrics from {metrics_file}: {e}")
        
        if metrics:
            results.append({
                'model': model,
                'input_output': input_output,
                'seq_len': int(seq_len),
                'pred_len': int(pred_len),
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'MSPE': metrics['MSPE']
            })
            print(f"✅ {model} ({input_output}): MAE={metrics['MAE']:.6f}")
    
    if not results:
        print("❌ No valid results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Custom sorting: 96->96, 96->192, 96->336, 96->720 (by pred_len)
    df = df.sort_values(['pred_len', 'MSE'])
    
    # Print metrics ranking for each experiment group (ordered by pred_len)
    print(f"\n📊 Experiment Group Rankings (by MSE - Low to High):")
    ordered_groups = sorted(df['input_output'].unique(), key=lambda x: int(x.split('->')[1]))
    for group in ordered_groups:
        group_df = df[df['input_output'] == group].sort_values('MSE')
        print(f"\n🎯 Group {group}:")
        print("-" * 80)
        for rank, (_, row) in enumerate(group_df.iterrows(), 1):
            print(f"{rank}. {row['model']:12s} | MSE: {row['MSE']:.6f} | MAE: {row['MAE']:.6f} | RMSE: {row['RMSE']:.6f} | MAPE: {row['MAPE']:.6f} | MSPE: {row['MSPE']:.2f}")
    
    # Prepare CSV output with groups and overall model means
    csv_rows = []
    csv_rows.append(['model', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    for group in ordered_groups:
        group_df = df[df['input_output'] == group].sort_values('MSE')
        
        # Add group data
        for _, row in group_df.iterrows():
            csv_rows.append([
                row['model'], row['seq_len'], row['pred_len'], 
                row['MSE'], row['MAE'], row['RMSE'], row['MAPE'], row['MSPE']
            ])
        
        # Add separator line (except for last group)
        if group != ordered_groups[-1]:
            csv_rows.append(['model', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    # Add overall model ranking means at the end
    csv_rows.append(['model', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    # Calculate overall model averages and sort by MSE
    model_avg_metrics = df.groupby('model')[['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']].mean().sort_values('MSE')
    
    for model, metrics in model_avg_metrics.iterrows():
        csv_rows.append([
            model, 'mean', 'mean',
            metrics['MSE'], metrics['MAE'], metrics['RMSE'], 
            metrics['MAPE'], metrics['MSPE']
        ])
    
    # Create DataFrame from csv_rows for output
    df_output = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
    
    # Save to CSV
    output_file = 'data_load_analysis.csv'
    df_output.to_csv(output_file, index=False)
    
    print(f"\n📊 Analysis Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Prediction horizons: {sorted(df['input_output'].unique())}")
    
    # Show overall model ranking by average MSE
    model_avg_mse = df.groupby('model')['MSE'].mean().sort_values()
    print(f"\n🏆 Overall Model Ranking (by average MSE):")
    for rank, (model, mse) in enumerate(model_avg_mse.items(), 1):
        print(f"{rank}. {model}: {mse:.6f}")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    analyze_load_data_results()
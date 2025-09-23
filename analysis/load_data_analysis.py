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

def analyze_load_data_results(granularity=5):
    """
    Analyze load_data model results and save to CSV
    
    Args:
        granularity (int): Time granularity in minutes (5, 15, 60). Default is 5.
    """
    
    print(f"🔍 Analyzing Load_Data Results (Granularity: {granularity} min)...")
    
    results_dir = Path('./results')
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    # Collect results
    results = []
    
    for folder in results_dir.iterdir():
        if not folder.is_dir() or 'load_data' not in folder.name:
            continue
        
        # Extract granularity from folder name
        folder_granularity = None
        if '_5min_' in folder.name:
            folder_granularity = 5
        elif '_15min_' in folder.name:
            folder_granularity = 15
        elif '_60min_' in folder.name:
            folder_granularity = 60
        else:
            # Default granularity if not specified in folder name
            folder_granularity = 5
        
        # Skip if granularity doesn't match the specified one
        if folder_granularity != granularity:
            continue
        
        # Parse folder name using regex
        try:
            # Updated pattern to handle granularity extraction
            if f'_{granularity}min_' in folder.name:
                pattern = r'long_term_forecast_load_data_\d+min_(\d+)_(\d+)_([^_]+)_'
            else:
                # For folders without granularity specification (default 5min)
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
                'granularity': folder_granularity,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'MSPE': metrics['MSPE']
            })
            print(f"✅ {model} ({input_output}, {folder_granularity}min): MAE={metrics['MAE']:.6f}")
    
    if not results:
        print(f"❌ No valid results found for {granularity}min granularity!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Custom sorting: 96->96, 96->192, 96->336, 96->720 (by pred_len)
    df = df.sort_values(['pred_len', 'MSE'])
    
    # Print metrics ranking for each experiment group (ordered by pred_len)
    print(f"\n📊 Experiment Group Rankings for {granularity}min (by MSE - Low to High):")
    ordered_groups = sorted(df['input_output'].unique(), key=lambda x: int(x.split('->')[1]))
    for group in ordered_groups:
        group_df = df[df['input_output'] == group].sort_values('MSE')
        print(f"\n🎯 Group {group} ({granularity}min):")
        print("-" * 80)
        for rank, (_, row) in enumerate(group_df.iterrows(), 1):
            print(f"{rank}. {row['model']:12s} | MSE: {row['MSE']:.6f} | MAE: {row['MAE']:.6f} | RMSE: {row['RMSE']:.6f} | MAPE: {row['MAPE']:.6f} | MSPE: {row['MSPE']:.2f}")
    
    # Prepare CSV output with groups and overall model means
    csv_rows = []
    csv_rows.append(['model', 'granularity', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    for group in ordered_groups:
        group_df = df[df['input_output'] == group].sort_values('MSE')
        
        # Add group data
        for _, row in group_df.iterrows():
            csv_rows.append([
                row['model'], granularity, row['seq_len'], row['pred_len'], 
                row['MSE'], row['MAE'], row['RMSE'], row['MAPE'], row['MSPE']
            ])
        
        # Add separator line (except for last group)
        if group != ordered_groups[-1]:
            csv_rows.append(['model', 'granularity', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    # Add overall model ranking means at the end
    csv_rows.append(['model', 'granularity', 'seq_len', 'pred_len', 'MSE', 'MAE', 'RMSE', 'MAPE', 'MSPE'])
    
    # Calculate overall model averages and sort by MSE
    model_avg_metrics = df.groupby('model')[['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']].mean().sort_values('MSE')
    
    for model, metrics in model_avg_metrics.iterrows():
        csv_rows.append([
            model, granularity, 'mean', 'mean',
            metrics['MSE'], metrics['MAE'], metrics['RMSE'], 
            metrics['MAPE'], metrics['MSPE']
        ])
    
    # Create DataFrame from csv_rows for output
    df_output = pd.DataFrame(csv_rows[1:], columns=csv_rows[0])
    
    # Save to CSV
    output_file = f'analysis/load_data_analysis_{granularity}min.csv'
    df_output.to_csv(output_file, index=False)
    
    print(f"\n📊 Analysis Summary ({granularity}min granularity):")
    print(f"Total experiments: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Prediction horizons: {sorted(df['input_output'].unique())}")
    
    # Show overall model ranking by average MSE
    model_avg_mse = df.groupby('model')['MSE'].mean().sort_values()
    print(f"\n🏆 Overall Model Ranking for {granularity}min (by average MSE):")
    for rank, (model, mse) in enumerate(model_avg_mse.items(), 1):
        print(f"{rank}. {model}: {mse:.6f}")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze load_data model results')
    parser.add_argument('--granularity', type=int, choices=[5, 15, 60], default=60,
                        help='Time granularity in minutes (5, 15, 60). Default is 5.')
    
    args = parser.parse_args()
    analyze_load_data_results(granularity=args.granularity)
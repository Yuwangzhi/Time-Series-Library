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
    
    # Sort by input_output and MAE for better readability
    df = df.sort_values(['input_output', 'MAE'])
    
    # Save to CSV
    output_file = 'data_load_analysis.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n📊 Analysis Summary:")
    print(f"Total experiments: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Prediction horizons: {sorted(df['input_output'].unique())}")
    
    # Show model ranking
    model_avg = df.groupby('model')['MAE'].mean().sort_values()
    print(f"\n🏆 Model Ranking (by average MAE):")
    for rank, (model, mae) in enumerate(model_avg.items(), 1):
        print(f"{rank}. {model}: {mae:.6f}")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    analyze_load_data_results()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import argparse
import os

def load_and_fit_scaler(train_data_path, target_column='value'):
    """Load training dataset and fit scaler"""
    print(f"Loading training dataset for scaler: {train_data_path}")
    df_raw = pd.read_csv(train_data_path, dtype={target_column: 'str'})
    
    # Handle non-numeric values
    df_raw[target_column] = pd.to_numeric(df_raw[target_column], errors='coerce')
    df_raw = df_raw.dropna()
    
    scaler = StandardScaler()
    scaler.fit(df_raw[[target_column]].values)
    
    print(f"Scaler fitted on {len(df_raw)} training samples")
    return scaler

def detect_mode_from_filename(prediction_csv_path):
    """Detect mode from prediction CSV filename"""
    filename = os.path.basename(prediction_csv_path).lower()
    if 'mode1' in filename:
        return 1
    elif 'mode2' in filename:
        return 2
    else:
        # Default fallback: check if file has 'true_value' column
        try:
            df = pd.read_csv(prediction_csv_path, nrows=1)
            if 'true_value' in df.columns:
                return 2
            else:
                return 1
        except:
            return 1  # Default to mode 1 if detection fails

def visualize_mode1(prediction_csv_path, train_data_path, 
                   input_len=96, output_path="prediction_mode1.png"):
    """
    Visualize Mode 1: Future prediction
    Orange: Input data, Blue: Predicted output data
    """
    # Load scaler
    scaler = load_and_fit_scaler(train_data_path)
    
    # Read prediction results first to get file structure
    print(f"Loading prediction results: {prediction_csv_path}")
    pred_df = pd.read_csv(prediction_csv_path)
    
    # Check if data_type column exists (new format)
    if 'data_type' in pred_df.columns:
        print("Mode 1: Using new CSV format with data_type column")
        
        # Separate input and prediction data
        input_data = pred_df[pred_df['data_type'] == 'input'].copy()
        pred_data = pred_df[pred_df['data_type'] == 'prediction'].copy()
        
        # Parse dates
        if 'timestamp' in input_data.columns:
            try:
                # Try to parse as datetime first
                input_dates = pd.to_datetime(input_data['timestamp']).tolist()
            except:
                # If parsing fails, create dummy dates
                start_date = datetime(2025, 1, 1, 0, 0)
                input_dates = [start_date + timedelta(hours=i) for i in range(len(input_data))]
        else:
            start_date = datetime(2025, 1, 1, 0, 0)
            input_dates = [start_date + timedelta(hours=i) for i in range(len(input_data))]
            
        if 'timestamp' in pred_data.columns:
            try:
                # Try to parse as datetime first
                pred_dates = pd.to_datetime(pred_data['timestamp']).tolist()
            except:
                # If parsing fails, create dummy dates continuing from input
                last_input_date = input_dates[-1] if input_dates else datetime(2025, 1, 1, 0, 0)
                pred_dates = [last_input_date + timedelta(hours=i+1) for i in range(len(pred_data))]
        else:
            last_input_date = input_dates[-1] if input_dates else datetime(2025, 1, 1, 0, 0)
            pred_dates = [last_input_date + timedelta(hours=i+1) for i in range(len(pred_data))]
        
        # Get values
        input_values = input_data['denormalized_value'].values
        pred_values = pred_data['denormalized_value'].values
        
    else:
        # Old format - split the CSV data based on input_len
        print("Mode 1: Using old CSV format")
        print(f"Total rows in CSV: {len(pred_df)}")
        
        # For old format, calculate output_len from total rows
        total_rows = len(pred_df)
        output_len = total_rows - input_len
        
        if output_len <= 0:
            raise ValueError(f"Invalid data: total rows ({total_rows}) must be greater than input_len ({input_len})")
        
        print(f"Calculated output_len: {output_len} (total_rows={total_rows} - input_len={input_len})")
        
        # Split data: first input_len rows are input, remaining rows are predictions
        input_data = pred_df.iloc[:input_len].copy()
        pred_data = pred_df.iloc[input_len:].copy()
        
        print(f"Split into {len(input_data)} input rows and {len(pred_data)} prediction rows")
        
        # Parse dates from CSV
        if 'timestamp' in pred_df.columns:
            try:
                all_dates = pd.to_datetime(pred_df['timestamp']).tolist()
                input_dates = all_dates[:input_len]
                pred_dates = all_dates[input_len:]
            except:
                # Create dummy dates
                start_date = datetime(2025, 1, 1, 0, 0)
                input_dates = [start_date + timedelta(hours=i) for i in range(input_len)]
                pred_dates = [start_date + timedelta(hours=input_len + i) for i in range(len(pred_data))]
        else:
            # Create dummy dates
            start_date = datetime(2025, 1, 1, 0, 0)
            input_dates = [start_date + timedelta(hours=i) for i in range(input_len)]
            pred_dates = [start_date + timedelta(hours=input_len + i) for i in range(len(pred_data))]
        
        # Get input values from input portion
        input_values = input_data['denormalized_value'].values
        
        # Get prediction values from prediction portion
        if 'denormalized_prediction' in pred_data.columns:
            pred_values = pred_data['denormalized_prediction'].values
        elif 'normalized_prediction' in pred_data.columns:
            pred_values_norm = pred_data['normalized_prediction'].values
            pred_values = scaler.inverse_transform(pred_values_norm.reshape(-1, 1)).flatten()
        elif 'denormalized_value' in pred_data.columns:
            pred_values = pred_data['denormalized_value'].values
        elif 'normalized_value' in pred_data.columns:
            pred_values_norm = pred_data['normalized_value'].values
            pred_values = scaler.inverse_transform(pred_values_norm.reshape(-1, 1)).flatten()
        else:
            raise ValueError("No prediction columns found in prediction file")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot input data (orange)
    plt.plot(input_dates, input_values, 'o-', color='orange', linewidth=2, 
             label=f'Input Data ({len(input_values)} steps)', alpha=0.8, markersize=4)
    
    # Plot predicted data (blue)
    plt.plot(pred_dates, pred_values, 's-', color='blue', linewidth=2, 
             label=f'Predicted Output ({len(pred_values)} steps)', alpha=0.8, markersize=4)
    
    # 分别连接输入区最后一个点和预测区第一个点（预测值和真实值）
    if len(input_dates) > 0 and len(pred_dates) > 0:
        # 连接预测值（蓝色）
        plt.plot(
            [input_dates[-1], pred_dates[0]],
            [input_values[-1], pred_values[0]],
            color='blue', linestyle='--', linewidth=2, alpha=0.8
        )

    # Add vertical line to separate input and prediction
    if len(input_dates) > 0 and len(pred_dates) > 0:
        separation_time = input_dates[-1] if input_dates else pred_dates[0] - timedelta(hours=1)
        plt.axvline(x=separation_time, color='black', linestyle='--', alpha=0.5, linewidth=1,
                    label='Prediction Start')
    
    # Formatting
    plt.title('Mode 1: Future Prediction Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Load Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Statistics
    print(f"\nMode 1 Statistics:")
    print(f"Input - Mean: {np.mean(input_values):.2f}, Std: {np.std(input_values):.2f}")
    print(f"Prediction - Mean: {np.mean(pred_values):.2f}, Std: {np.std(pred_values):.2f}")
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mode 1 visualization saved to: {output_path}")
    plt.show()

def visualize_mode2(prediction_csv_path, train_data_path, 
                   input_len=96, output_len=96, output_path="prediction_mode2.png"):
    """
    Visualize Mode 2: Historical sliding window prediction
    Orange: Input data, Blue: Predicted output, Green: True values
    """
    # Load scaler
    scaler = load_and_fit_scaler(train_data_path)
    
    # Read prediction results first to get structure and data
    print(f"Loading prediction results: {prediction_csv_path}")
    pred_df = pd.read_csv(prediction_csv_path)
    
    print("Mode 2: Using new CSV format with data_type column")
    
    # Separate input and prediction data
    input_data = pred_df[pred_df['data_type'] == 'input'].copy()
    pred_data = pred_df[pred_df['data_type'] == 'prediction'].copy()
    
    print(f"Found {len(input_data)} input rows and {len(pred_data)} prediction rows")
    
    # Parse dates
    if 'timestamp' in input_data.columns:
        try:
            # Try to parse as datetime first
            input_dates = pd.to_datetime(input_data['timestamp']).tolist()
        except:
            # If parsing fails, create dummy dates
            start_date = datetime(2025, 1, 1, 0, 0)
            input_dates = [start_date + timedelta(hours=i) for i in range(len(input_data))]
    else:
        start_date = datetime(2025, 1, 1, 0, 0)
        input_dates = [start_date + timedelta(hours=i) for i in range(len(input_data))]
        
    if 'timestamp' in pred_data.columns:
        try:
            # Try to parse as datetime first
            pred_dates = pd.to_datetime(pred_data['timestamp']).tolist()
            true_dates = pred_dates  # Same dates for true values
        except:
            # If parsing fails, create dummy dates continuing from input
            last_input_date = input_dates[-1] if input_dates else datetime(2025, 1, 1, 0, 0)
            pred_dates = [last_input_date + timedelta(hours=i+1) for i in range(len(pred_data))]
            true_dates = pred_dates
    else:
        last_input_date = input_dates[-1] if input_dates else datetime(2025, 1, 1, 0, 0)
        pred_dates = [last_input_date + timedelta(hours=i+1) for i in range(len(pred_data))]
        true_dates = pred_dates
    
    # Get values
    input_values = input_data['denormalized_value'].values
    pred_values = pred_data['denormalized_value'].values
    true_values = pred_data['true_value'].values
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot input data (orange)
    plt.plot(input_dates, input_values, 'o-', color='orange', linewidth=2, 
             label=f'Input Data ({input_len} steps)', alpha=0.8, markersize=4)
    
    # Plot predicted data (blue)
    plt.plot(pred_dates, pred_values, 's-', color='blue', linewidth=2, 
             label=f'Predicted Output ({len(pred_values)} steps)', alpha=0.8, markersize=4)
    
    # Plot true values (green)
    plt.plot(true_dates, true_values, '^-', color='green', linewidth=2, 
             label=f'True Values ({len(true_values)} steps)', alpha=0.8, markersize=4)
    
    # 分别连接输入区最后一个点和预测区第一个点（预测值和真实值）
    if len(input_dates) > 0 and len(pred_dates) > 0:
        # 连接预测值（蓝色）
        plt.plot(
            [input_dates[-1], pred_dates[0]],
            [input_values[-1], pred_values[0]],
            color='blue', linestyle='--', linewidth=2, alpha=0.8
        )
        # 连接真实值（绿色）
        plt.plot(
            [input_dates[-1], true_dates[0]],
            [input_values[-1], true_values[0]],
            color='green', linestyle='--', linewidth=2, alpha=0.8
        )

    # Add vertical line to separate input and prediction
    if len(input_dates) > 0 and len(true_dates) > 0:
        plt.axvline(x=input_dates[-1], color='black', linestyle='--', alpha=0.5, linewidth=1,
                    label='Prediction Start')
    
    # Formatting
    plt.title('Mode 2: Historical Sliding Window Prediction Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Load Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Statistics and metrics
    print(f"\nMode 2 Statistics:")
    print(f"Input - Mean: {np.mean(input_values):.2f}, Std: {np.std(input_values):.2f}")
    print(f"Prediction - Mean: {np.mean(pred_values):.2f}, Std: {np.std(pred_values):.2f}")
    print(f"True Values - Mean: {np.mean(true_values):.2f}, Std: {np.std(true_values):.2f}")
    
    # Calculate metrics
    if len(pred_values) == len(true_values):
        mae = np.mean(np.abs(pred_values - true_values))
        mse = np.mean((pred_values - true_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
        
        print(f"\nPrediction Metrics:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")
    
    # Save and show
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Mode 2 visualization saved to: {output_path}")
    plt.show()


import re

def extract_lens_from_filename(filename):
    """从文件名中自动提取input_len和output_len"""
    # 支持 inXXX_outYYY 或 inXXX_outYYY_gZ 这样的命名
    basename = os.path.basename(filename)
    m = re.search(r'_in(\d+)_out(\d+)', basename)
    if m:
        input_len = int(m.group(1))
        output_len = int(m.group(2))
        return input_len, output_len
    # fallback
    return None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize predictions from predict_.py')
    parser.add_argument('--prediction_csv_path', type=str, required=False,
                        help='Path to prediction results CSV file', # ['Informer', 'PatchTST', 'DLinear']
                        default=r'./predict/output/predictions_Informer_in336_out96_60min_mode2_start-1_20250720-20250807.csv')
    parser.add_argument('--train_data_path', type=str, required=False,
                        help='Path to training dataset for fitting scaler', 
                        default=r'./dataset/load_data/hf_load_data/hf_load_data_20210101-20250807_60min.csv')
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to save visualization (default: auto)', 
                        default=None)

    args = parser.parse_args()

    # 自动提取input_len和output_len
    input_len, output_len = extract_lens_from_filename(args.prediction_csv_path)
    if input_len is None or output_len is None:
        raise ValueError('Cannot extract input_len/output_len from prediction_csv_path filename!')

    # Auto-detect mode from filename
    mode = detect_mode_from_filename(args.prediction_csv_path)

    # Validate file existence
    if not os.path.exists(args.prediction_csv_path):
        raise FileNotFoundError(f"Prediction CSV file not found: {args.prediction_csv_path}")
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data file not found: {args.train_data_path}")

    # 自动生成output_path（与csv同名，仅后缀为png）
    if args.output_path is None:
        base = os.path.splitext(os.path.basename(args.prediction_csv_path))[0]
        args.output_path = os.path.join(os.path.dirname(args.prediction_csv_path), base + '.png')

    print("="*80)
    print("Prediction Visualization Tool (Auto Mode Detection)")
    print("="*80)
    print(f"Prediction CSV: {args.prediction_csv_path}")
    print(f"Training Data: {args.train_data_path}")
    print(f"Auto-detected Mode: {mode} ({'Future Prediction' if mode == 1 else 'Historical Sliding Window'})")
    print(f"Input Length: {input_len}")
    print(f"Output Length: {output_len}")
    print(f"Output Path: {args.output_path}")
    print("="*80)

    # Run visualization based on auto-detected mode
    if mode == 1:
        visualize_mode1(
            args.prediction_csv_path, 
            args.train_data_path,
            input_len,
            args.output_path
        )
    elif mode == 2:
        visualize_mode2(
            args.prediction_csv_path, 
            args.train_data_path,
            input_len,
            output_len,
            args.output_path
        )

    print("="*80)
    print("Visualization completed successfully!")
    print("="*80)
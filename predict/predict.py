import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

# from models.Informer import Model
# from models.PatchTST import Model
from utils.timefeatures import time_features


class PredictionDataset(Dataset):
    """Custom dataset for prediction from CSV file"""

    def __init__(self, data, seq_len, label_len, pred_len, time_stamps=None, mode=1, start_idx=-1):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.time_stamps = time_stamps
        self.mode = mode  # 1: future prediction, 2: historical prediction (无未来真实值)
        self.start_idx = start_idx

        if mode == 1:
            self.required_len = self.seq_len
            if start_idx == -1:
                self.actual_start = len(self.data) - self.required_len
            else:
                self.actual_start = start_idx
        else:
            # mode 2: 向前取 output_len 个点，模拟没有未来真实值
            self.required_len = self.seq_len
            if start_idx == -1:
                self.actual_start = len(self.data) - self.seq_len - self.pred_len
            else:
                self.actual_start = start_idx - self.pred_len

        if self.actual_start < 0:
            raise ValueError(f"Not enough data. Need {self.required_len} samples, but start_idx {start_idx} would require data before beginning.")
        if self.actual_start + self.required_len + (self.pred_len if mode == 2 else 0) > len(self.data):
            raise ValueError(f"Not enough data. Need {self.required_len + (self.pred_len if mode == 2 else 0)} samples from index {self.actual_start}, but only have {len(self.data)} total samples.")

        print(f"Dataset created: Mode {mode}, Start index: {self.actual_start}, Required length: {self.required_len}")

    def __len__(self):
        return 1  # mode 1 和 mode 2 都只预测一次

    def __getitem__(self, index):
        s_begin = self.actual_start
        s_end = self.actual_start + self.seq_len

        seq_x = self.data[s_begin:s_end]

        if self.mode == 1:
            r_begin = s_end - self.label_len
            seq_y = np.zeros((self.label_len + self.pred_len, seq_x.shape[-1]))
            seq_y[:self.label_len] = self.data[r_begin:s_end]
        else:
            r_begin = s_end - self.label_len
            seq_y = np.zeros((self.label_len + self.pred_len, seq_x.shape[-1]))
            seq_y[:self.label_len] = self.data[r_begin:s_end]
            # 未来部分填零，模型预测

        if self.time_stamps is not None:
            seq_x_mark = self.time_stamps[s_begin:s_end]
            seq_y_mark = np.zeros((self.label_len + self.pred_len, self.time_stamps.shape[-1]))
            seq_y_mark[:self.label_len] = self.time_stamps[r_begin:s_end]
            if s_end + self.pred_len <= len(self.time_stamps):
                seq_y_mark[self.label_len:] = self.time_stamps[s_end:s_end + self.pred_len]
            else:
                for i in range(self.pred_len):
                    seq_y_mark[self.label_len + i] = seq_y_mark[self.label_len - 1]
        else:
            seq_x_mark = np.zeros((self.seq_len, 4))
            seq_y_mark = np.zeros((self.label_len + self.pred_len, 4))

        return (
            torch.FloatTensor(seq_x),
            torch.FloatTensor(seq_y),
            torch.FloatTensor(seq_x_mark),
            torch.FloatTensor(seq_y_mark),
            index
        )

def load_and_fit_scaler(train_data_path, target_column='value', train_ratio=0.7):
    """Load training dataset and fit scaler using only first 70% of data"""
    print(f"Loading training dataset from: {train_data_path}")
    df_raw = pd.read_csv(train_data_path, dtype={target_column: 'str'})  # Read as string first
    
    # Handle non-numeric values in target column
    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in training dataset. Available columns: {df_raw.columns.tolist()}")
    
    # Replace 'noval' and other non-numeric values with NaN
    df_raw[target_column] = pd.to_numeric(df_raw[target_column], errors='coerce')
    
    # Remove rows with NaN values (including converted 'noval' entries)
    df_raw = df_raw.dropna()
    
    print(f"After cleaning non-numeric values, {len(df_raw)} training samples remain")
    
    # Extract target feature data
    df_data = df_raw[[target_column]]
    
    # Use only the first 70% of data for fitting scaler
    total_samples = len(df_data)
    train_samples = int(total_samples * train_ratio)
    train_data = df_data.iloc[:train_samples]
    
    print(f"Total samples: {total_samples}")
    print(f"Using first {train_ratio*100:.0f}% ({train_samples}) samples for fitting scaler")
    
    # Fit scaler on training portion only
    scaler = StandardScaler()
    scaler.fit(train_data.values)
    
    print(f"Scaler fitted on {len(train_data)} training samples")
    print(f"Scaler mean: {scaler.mean_[0]:.4f}, std: {scaler.scale_[0]:.4f}")
    
    return scaler


def load_input_data(input_csv_path, scaler, target_column='value', with_time=True):
    """Load input CSV and apply normalization using training scaler"""
    print(f"Loading input data from: {input_csv_path}")
    df_raw = pd.read_csv(input_csv_path, dtype={target_column: 'str'})  # Read as string first
    
    # Handle non-numeric values in target column
    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in input dataset. Available columns: {df_raw.columns.tolist()}")
    
    # Replace 'noval' and other non-numeric values with NaN
    df_raw[target_column] = pd.to_numeric(df_raw[target_column], errors='coerce')
    
    # Remove rows with NaN values (including converted 'noval' entries)
    df_raw = df_raw.dropna()
    
    print(f"After cleaning non-numeric values, {len(df_raw)} samples remain")
    
    # Extract target feature data
    df_data = df_raw[[target_column]]
    
    # Apply normalization using training scaler
    data_normalized = scaler.transform(df_data.values)
    
    # Process time features if available
    time_stamps = None
    dates = None
    if with_time and 'date' in df_raw.columns:
        df_stamp = df_raw[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        dates = df_stamp['date'].values
        
        # Use timeF encoding (same as training)
        time_stamps = time_features(pd.to_datetime(df_stamp['date'].values), freq='h')
        time_stamps = time_stamps.transpose(1, 0)  # Shape: (n_samples, n_features)
    
    print(f"Input data loaded: {len(data_normalized)} samples")
    return data_normalized, time_stamps, dates, df_data.values  # Return original values for comparison


def create_informer_config(seq_len, label_len, pred_len):
    """Create configuration object for Informer model matching training params"""
    class Config:
        def __init__(self):
            # Task settings
            self.task_name = 'long_term_forecast'
            self.features = 'S'  # Univariate
            self.target = 'value'
            
            # Model architecture (matching training script exactly)
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            self.enc_in = 1  # Input features
            self.dec_in = 1  # Decoder input features
            self.c_out = 1   # Output features
            self.d_model = 512
            self.n_heads = 8
            self.e_layers = 3
            self.d_layers = 1
            self.d_ff = 2048
            self.factor = 3
            self.distil = True  # Default is True
            self.dropout = 0.1
            self.activation = 'gelu'
            
            # Embedding settings
            self.embed = 'timeF'  # Time features encoding
            self.freq = 'h'       # Hourly frequency
            
    return Config()

def create_patchtst_config(seq_len, label_len, pred_len):
    """Create configuration object for PatchTST model matching training params"""
    class Config:
        def __init__(self):
            # Task settings
            self.task_name = 'long_term_forecast'
            self.features = 'S'  # Univariate
            self.target = 'value'

            # PatchTST architecture
            self.seq_len = seq_len
            self.label_len = label_len
            self.pred_len = pred_len
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
            self.d_model = 512
            self.n_heads = 4
            self.e_layers = 1
            self.d_layers = 1
            self.d_ff = 2048
            self.factor = 3
            self.dropout = 0.1
            self.activation = 'gelu'
            self.embed = 'timeF'
            self.freq = 'h'
            # PatchTST-specific
            self.patch_len = 16
            self.stride = 8

    return Config()

def load_model_checkpoint(model, ckpt_path, device):
    """Load model weights from checkpoint"""
    print(f"Loading model checkpoint from: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    print("Model checkpoint loaded successfully")
    return model


def generate_future_timestamps(last_timestamp, steps, freq='h'):
    """Generate future timestamps for prediction"""
    if isinstance(last_timestamp, str):
        last_timestamp = pd.to_datetime(last_timestamp)
    
    freq_map = {'h': 'H', 'd': 'D', 'm': 'T'}
    pd_freq = freq_map.get(freq, 'H')
    
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1), 
        periods=steps, 
        freq=pd_freq
    )
    
    return future_timestamps


def predict_from_csv(args):
    """Main prediction function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    scaler = load_and_fit_scaler(args.train_data_path, args.target)
    input_data, time_stamps, dates, original_values = load_input_data(args.input_csv_path, scaler, args.target)

    print(f"Total input data samples: {len(input_data)}")


    # 根据 model_name 选择模型和配置
    model_name = getattr(args, 'model_name', 'PatchTST')
    if model_name.lower() == 'informer':
        from models.Informer import Model as InformerModel
        config = create_informer_config(args.input_len, args.label_len, args.output_len)
        model = InformerModel(config).to(device)
        model_type_str = 'Informer'
        nh = '8'
        el = '3'
    else:
        from models.PatchTST import Model as PatchTSTModel
        config = create_patchtst_config(args.input_len, args.label_len, args.output_len)
        model = PatchTSTModel(config).to(device)
        model_type_str = 'PatchTST'
        nh = '4'
        el = '1'

    # 自动查找ckpt
    ckpt_path = getattr(args, 'ckpt_path', None)
    if not ckpt_path or not os.path.exists(ckpt_path):
        ckpt_dir = f"checkpoints/long_term_forecast_load_data_60min_{args.input_len}_{args.output_len}_{model_type_str}_load_data_ftS_sl{args.input_len}_ll{args.label_len}_pl{args.output_len}_dm512_nh{nh}_el{el}_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
        ckpt_path_candidate = os.path.join(ckpt_dir, "checkpoint.pth")
        if not os.path.exists(ckpt_path_candidate):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path or ckpt_path_candidate}\nTried auto path: {ckpt_path_candidate}")
        print(f"Auto-selected checkpoint: {ckpt_path_candidate}")
        ckpt_path = ckpt_path_candidate
    else:
        print(f"Using user-specified checkpoint: {ckpt_path}")

    model = load_model_checkpoint(model, ckpt_path, device)
    model.eval()

    dataset = PredictionDataset(
        input_data,
        args.input_len,
        args.label_len,
        args.output_len,
        time_stamps,
        args.mode,
        args.start_idx
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = []
    prediction_indices = []

    print(f"Starting prediction in mode {args.mode}...")
    with torch.no_grad():
        for batch_data in dataloader:
            batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch_data

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.output_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).to(device)
            extract_len = args.output_len

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred_output = outputs[:, -extract_len:, :].detach().cpu().numpy()

            predictions.append(pred_output)
            prediction_indices.append(index.item() if hasattr(index, 'item') else index)

    predictions = np.concatenate(predictions, axis=0)
    print(f"Prediction completed. Shape: {predictions.shape}")

    if args.mode == 1:
        results = prepare_future_results_with_start_idx(predictions, dates, scaler, args.output_len, dataset.actual_start, args.input_len, input_data, original_values)
    else:
        results = prepare_historical_results_with_start_idx(predictions, dates, original_values, scaler, args.output_len, dataset.actual_start, args.input_len, input_data)

    # 提取区间日期（优先用 prediction 区间的时间戳）
    input_rows = results[results['data_type'] == 'input']
    pred_rows = results[results['data_type'] == 'prediction']

    def try_parse_date(s):
        try:
            return pd.to_datetime(s)
        except:
            return None

    # 推断预测区间的日期范围用于文件命名
    if dates is not None and len(dates) > 0:
        if args.mode == 1:
            # mode 1: 真实预测
            if args.start_idx == -1:
                # 使用最后的 input_len 作为输入，预测未来 output_len
                input_start_idx = len(dates) - args.input_len
                pred_start_date = pd.to_datetime(dates[-1]) + pd.Timedelta(hours=1)
                pred_end_date = pred_start_date + pd.Timedelta(hours=args.output_len - 1)
            else:
                # 从指定 start_idx 开始的 input_len 作为输入，预测 output_len
                input_start_idx = args.start_idx
                pred_start_date = pd.to_datetime(dates[args.start_idx + args.input_len])
                pred_end_date = pd.to_datetime(dates[args.start_idx + args.input_len + args.output_len - 1])
            
            # 文件命名用输入区间的起始日期到预测区间的结束日期
            start_date = pd.to_datetime(dates[input_start_idx]).strftime("%Y%m%d")
            end_date = pred_end_date.strftime("%Y%m%d")
            date_range_str = f"{start_date}-{end_date}"
            
        else:
            # mode 2: 模拟预测
            if args.start_idx == -1:
                # 使用倒数第 (input_len + output_len) 到倒数第 output_len 作为输入
                input_start_idx = len(dates) - args.input_len - args.output_len
                pred_start_idx = len(dates) - args.output_len
                pred_end_idx = len(dates) - 1
            else:
                # 从 (start_idx - output_len) 开始的 input_len 作为输入
                input_start_idx = args.start_idx - args.output_len
                pred_start_idx = args.start_idx
                pred_end_idx = args.start_idx + args.output_len - 1
            
            # 文件命名用输入区间的起始日期到预测区间的结束日期
            start_date = pd.to_datetime(dates[input_start_idx]).strftime("%Y%m%d")
            end_date = pd.to_datetime(dates[pred_end_idx]).strftime("%Y%m%d")
            date_range_str = f"{start_date}-{end_date}"
    else:
        # 没有时间戳，用步数标记
        if args.mode == 1:
            date_range_str = f"Future{args.output_len}steps"
        else:
            date_range_str = f"Historical{args.output_len}steps"


    model_name = getattr(args, 'model_name', 'UnknownModel')
    granularity = getattr(args, 'granularity', 60)
    output_file = os.path.join(
        './predict/output',
        # os.path.dirname(args.input_csv_path),
        f'predictions_{model_name}_in{args.input_len}_out{args.output_len}_{granularity}min_mode{args.mode}_start{args.start_idx}_{date_range_str}.csv'
    )
    results.to_csv(output_file, index=False)

    print(f"Results saved to: {output_file}")
    print("\nFirst 10 rows of results:")
    print(results.head(10))

    return results

def prepare_future_results_with_start_idx(predictions, dates, scaler, output_len, actual_start, input_len, input_data, original_values):
    """Prepare results for mode 1 (future prediction)"""
    
    input_start_idx = actual_start
    input_end_idx = actual_start + input_len
    
    # 使用真实的输入数据日期
    if dates is not None and len(dates) >= input_end_idx:
        input_dates = dates[input_start_idx:input_end_idx]
        input_time_column = [pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in input_dates]
        # 获取最后一个输入时间戳用于生成预测时间戳
        last_input_time = pd.to_datetime(input_dates[-1])
    else:
        raise ValueError(f"Not enough date data. Need dates up to index {input_end_idx}, but only have {len(dates) if dates is not None else 0} dates.")

    input_normalized = input_data[input_start_idx:input_end_idx, 0]
    input_real = scaler.inverse_transform(input_normalized.reshape(-1, 1)).flatten()

    input_df = pd.DataFrame({
        'timestamp': input_time_column,
        'normalized_value': input_normalized,
        'denormalized_value': input_real,
        'data_type': 'input'
    })

    # 生成预测区间的时间戳（从最后一个输入时间的下一小时开始）
    pred_time_column = [(last_input_time + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S') for i in range(output_len)]

    pred_normalized = predictions[0, :, 0]
    pred_real = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()

    pred_df = pd.DataFrame({
        'timestamp': pred_time_column,
        'normalized_value': pred_normalized,
        'denormalized_value': pred_real,
        'data_type': 'prediction'
    })

    results = pd.concat([input_df, pred_df], ignore_index=True)
    return results

def prepare_historical_results_with_start_idx(predictions, dates, original_values, scaler, output_len, actual_start, input_len, input_data):
    """Prepare results for mode 2 (historical prediction,可对比真实值)"""

    input_start_idx = actual_start
    input_end_idx = actual_start + input_len

    # 使用真实的输入数据日期
    if dates is not None and len(dates) >= input_end_idx:
        input_dates = dates[input_start_idx:input_end_idx]
        input_time_column = [pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in input_dates]
    else:
        raise ValueError(f"Not enough date data. Need dates up to index {input_end_idx}, but only have {len(dates) if dates is not None else 0} dates.")

    input_normalized = input_data[input_start_idx:input_end_idx, 0]
    input_real = scaler.inverse_transform(input_normalized.reshape(-1, 1)).flatten()

    input_df = pd.DataFrame({
        'timestamp': input_time_column,
        'normalized_value': input_normalized,
        'denormalized_value': input_real,
        'true_value': original_values[input_start_idx:input_end_idx, 0],
        'data_type': 'input'
    })

    # 使用真实的预测区间日期
    pred_start_idx = input_end_idx
    pred_end_idx = pred_start_idx + output_len

    if dates is not None and len(dates) >= pred_end_idx:
        pred_dates = dates[pred_start_idx:pred_end_idx]
        pred_time_column = [pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in pred_dates]
    else:
        raise ValueError(f"Not enough date data for prediction period. Need dates up to index {pred_end_idx}, but only have {len(dates) if dates is not None else 0} dates.")

    pred_normalized = predictions[0, :, 0]
    pred_real = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()

    # 预测区间真实值
    true_values = original_values[pred_start_idx:pred_end_idx, 0]

    pred_df = pd.DataFrame({
        'timestamp': pred_time_column,
        'normalized_value': pred_normalized,
        'denormalized_value': pred_real,
        'true_value': true_values,
        'data_type': 'prediction'
    })

    results = pd.concat([input_df, pred_df], ignore_index=True)
    return results
def prepare_future_results(predictions, dates, scaler, output_len):
    """Prepare results for mode 1 (future prediction)"""
    # Generate future timestamps
    if dates is not None and len(dates) > 0:
        last_date = dates[-1]
        future_dates = generate_future_timestamps(last_date, output_len)
        time_column = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in future_dates]
    else:
        time_column = [f"Future_Step_{i+1}" for i in range(output_len)]
    
    # Extract predictions (assuming single batch)
    pred_normalized = predictions[0, :, 0]  # Shape: (output_len,)
    
    # Inverse transform to get real values
    pred_real = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'timestamp': time_column,
        'normalized_prediction': pred_normalized,
        'denormalized_prediction': pred_real
    })
    
    return results


def prepare_historical_results(predictions, dates, original_values, scaler, output_len):
    """Prepare results for mode 2 (historical sliding window)"""
    # Get the timestamps for the prediction period
    if dates is not None and len(dates) >= output_len:
        pred_dates = dates[-output_len:]
        time_column = [pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S') for ts in pred_dates]
    else:
        time_column = [f"Historical_Step_{i+1}" for i in range(output_len)]
    
    # Extract predictions (each row is one prediction)
    pred_normalized = predictions[:, 0, 0]  # Shape: (output_len,)
    
    # Inverse transform to get real values  
    pred_real = scaler.inverse_transform(pred_normalized.reshape(-1, 1)).flatten()
    
    # Get true values for comparison
    true_values = original_values[-output_len:, 0]  # Original values for the same period
    
    # Create results DataFrame
    results = pd.DataFrame({
        'timestamp': time_column,
        'normalized_prediction': pred_normalized,
        'denormalized_prediction': pred_real,
        'true_value': true_values
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Predict from CSV using trained model')
    
    # Required arguments
    parser.add_argument('--model_name', type=str, required=False,
                        help='Model name: Informer or PatchTST', 
                        choices=['Informer', 'PatchTST'], default='PatchTST')
    parser.add_argument('--granularity', type=int, required=False, default=60, choices=[5, 15, 60],
                        help='Time granularity in minutes (5, 15, 60). Default is 60.')
    parser.add_argument('--input_csv_path', type=str, required=False,
                        help='Path to input CSV file for prediction', 
                        default=r'./predict/data/hf_load_data_20210101-20250807_60min.csv')
    parser.add_argument('--train_data_path', type=str, required=False,
                        help='Path to training dataset for fitting scaler', 
                        default=r'./dataset/load_data/hf_load_data/hf_load_data_20210101-20250807_60min.csv')
    parser.add_argument('--ckpt_path', type=str, required=False,
                        help='Path to model checkpoint', 
                        # default=r'./checkpoints/long_term_forecast_load_data_60min_96_96_Informer_load_data_ftS_sl96_ll48_pl96_dm512_nh8_el3_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth')
                        # default=r'./checkpoints/long_term_forecast_load_data_60min_336_96_PatchTST_load_data_ftS_sl336_ll48_pl96_dm512_nh4_el1_dl1_df2048_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth')
    )
    parser.add_argument('--input_len', type=int, required=False,
                        help='Input sequence length (e.g., 96)', default=336)
    parser.add_argument('--label_len', type=int, required=False,
                        help='Label length (e.g., 48)', default=48)
    parser.add_argument('--output_len', type=int, required=False,
                        help='Prediction length (e.g., 720)', default=96)
    parser.add_argument('--mode', type=int, required=False, choices=[1, 2],
                        help='Prediction mode: 1=future prediction, 2=historical sliding window', default=1)
    parser.add_argument('--target', type=str, default='value',
                        help='Target column name')
    parser.add_argument('--start_idx', type=int, required=False, default=-1,
                        help='Starting index for data extraction. -1 means use last available data. '
                             'For mode 1: takes input_len from start_idx. '
                             'For mode 2: takes input_len+output_len from start_idx.')
    
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_csv_path):
        raise FileNotFoundError(f"Input CSV file not found: {args.input_csv_path}")
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training dataset not found: {args.train_data_path}")
    # if not os.path.exists(args.ckpt_path):
        # raise FileNotFoundError(f"Checkpoint file not found: {args.ckpt_path}")
    
    print("="*80)
    print("CSV Prediction with Informer Model (Enhanced Version)")
    print("="*80)
    print(f"Input CSV Path: {args.input_csv_path}")
    print(f"Training Data Path: {args.train_data_path}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Input Length: {args.input_len}")
    print(f"Label Length: {args.label_len}")
    print(f"Output Length: {args.output_len}")
    print(f"Prediction Mode: {args.mode} ({'Future Prediction' if args.mode == 1 else 'Historical Sliding Window'})")
    print(f"Start Index: {args.start_idx} ({'Last available data' if args.start_idx == -1 else f'From row {args.start_idx}'})")
    print(f"Target Column: {args.target}")
    print("="*80)
    
    # Run prediction
    results = predict_from_csv(args)
    
    print("="*80)
    print("Prediction completed successfully!")
    print("="*80)


if __name__ == '__main__':
    main()
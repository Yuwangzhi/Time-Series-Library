import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import argparse
import os
import re

DEFAULT_START_DATE = datetime(2025, 1, 1, 0, 0)


def _generate_dummy_dates(length, start=DEFAULT_START_DATE, freq_hours=1):
    """Create a list of equally spaced datetime objects when no timestamp exists."""
    return [start + timedelta(hours=i * freq_hours) for i in range(length)]


def _format_time_axis(ax, hour_interval=12, date_fmt='%m/%d %H:%M', rotation=45):
    """Apply a consistent datetime formatting style to the x-axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    ax.tick_params(axis='x', rotation=rotation)


def _split_by_data_type(df, required=('input', 'prediction')):
    """Split dataframe rows according to `data_type` and assert required presence."""
    if 'data_type' not in df.columns:
        raise ValueError("Expected 'data_type' column in prediction file for this mode.")

    groups = {key: grp.copy() for key, grp in df.groupby('data_type')}
    missing = [name for name in required if name not in groups or groups[name].empty]
    if missing:
        raise ValueError(f"Missing data_type entries: {', '.join(missing)}")
    return groups


def _resolve_values(df, scaler, primary_cols, normalized_cols=()):
    """Return a value series, inverse transforming when only normalized columns exist."""
    for col in primary_cols:
        if col in df.columns:
            return df[col].astype(float).values

    for col in normalized_cols:
        if col in df.columns:
            values = df[col].astype(float).values.reshape(-1, 1)
            if scaler is None:
                raise ValueError(f"Scaler required to inverse-transform column '{col}'.")
            return scaler.inverse_transform(values).flatten()

    raise ValueError(
        f"None of the expected columns found. Tried primary={primary_cols}, normalized={normalized_cols}."
    )

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

def _parse_dates(df_segment):
    """Parse dates from a dataframe segment"""
    if 'timestamp' in df_segment.columns:
        try:
            # Try to parse as datetime first
            parsed = pd.to_datetime(df_segment['timestamp'], errors='coerce')
            if parsed.notna().all():
                return parsed.tolist()
        except Exception:
            pass

    # Fallback to synthetic dates when parsing fails
    return _generate_dummy_dates(len(df_segment))

def _connect_points(plt_obj, x_points, y_points, colors):
    """Connect last input point with first prediction/true points"""
    if len(x_points) < 2:
        return
        
    last_input_x, last_input_y = x_points[0][-1], y_points[0][-1]
    
    for xs, ys, color in zip(x_points[1:], y_points[1:], colors[1:]):
        if len(xs) > 0:
            plt_obj.plot(
                [last_input_x, xs[0]],
                [last_input_y, ys[0]],
                color=color, linestyle='--', linewidth=2, alpha=0.8
            )


def _prepare_mode1_series(pred_df, scaler, input_len):
    """Return dates and values for mode 1, supporting both legacy and new formats."""
    if 'data_type' in pred_df.columns:
        groups = _split_by_data_type(pred_df, required=('input', 'prediction'))
        input_data = groups['input']
        pred_data = groups['prediction']

        input_dates = _parse_dates(input_data)
        pred_dates = _parse_dates(pred_data)

        input_values = _resolve_values(
            input_data,
            scaler,
            primary_cols=('denormalized_value',),
            normalized_cols=('normalized_value',)
        )
        pred_values = _resolve_values(
            pred_data,
            scaler,
            primary_cols=('denormalized_value', 'denormalized_prediction'),
            normalized_cols=('normalized_value', 'normalized_prediction')
        )
        return input_dates, input_values, pred_dates, pred_values

    # Legacy path: first `input_len` rows are input, remaining are predictions
    total_rows = len(pred_df)
    output_len = total_rows - input_len
    if output_len <= 0:
        raise ValueError(
            f"Invalid data: total rows ({total_rows}) must exceed input_len ({input_len})."
        )

    input_data = pred_df.iloc[:input_len].copy()
    pred_data = pred_df.iloc[input_len:].copy()

    if 'timestamp' in pred_df.columns:
        try:
            all_dates = pd.to_datetime(pred_df['timestamp'], errors='coerce')
            if all_dates.notna().all():
                input_dates = all_dates.iloc[:input_len].tolist()
                pred_dates = all_dates.iloc[input_len:].tolist()
            else:
                raise ValueError("Timestamp parsing produced NaT values")
        except Exception:
            input_dates = _generate_dummy_dates(input_len)
            pred_dates = _generate_dummy_dates(len(pred_data), start=input_dates[-1] + timedelta(hours=1))
    else:
        input_dates = _generate_dummy_dates(input_len)
        pred_dates = _generate_dummy_dates(len(pred_data), start=input_dates[-1] + timedelta(hours=1))

    input_values = _resolve_values(
        input_data,
        scaler,
        primary_cols=('denormalized_value',),
        normalized_cols=('normalized_value',)
    )
    pred_values = _resolve_values(
        pred_data,
        scaler,
        primary_cols=('denormalized_prediction', 'denormalized_value'),
        normalized_cols=('normalized_prediction', 'normalized_value')
    )

    return input_dates, input_values, pred_dates, pred_values


def _prepare_mode2_series(pred_df, scaler):
    """Return dates and values for mode 2 visualization."""
    groups = _split_by_data_type(pred_df, required=('input', 'prediction'))
    input_data = groups['input']
    pred_data = groups['prediction']

    input_dates = _parse_dates(input_data)
    pred_dates = _parse_dates(pred_data)
    true_dates = pred_dates

    input_values = _resolve_values(
        input_data,
        scaler,
        primary_cols=('denormalized_value',),
        normalized_cols=('normalized_value',)
    )
    pred_values = _resolve_values(
        pred_data,
        scaler,
        primary_cols=('denormalized_value', 'denormalized_prediction'),
        normalized_cols=('normalized_value', 'normalized_prediction')
    )

    if 'true_value' in pred_data.columns:
        true_values = pred_data['true_value'].astype(float).values
    elif 'true_value_normalized' in pred_data.columns:
        true_values = scaler.inverse_transform(
            pred_data['true_value_normalized'].astype(float).values.reshape(-1, 1)
        ).flatten()
    else:
        raise ValueError("Mode 2 requires a 'true_value' (or normalized) column in prediction data.")

    return input_dates, input_values, pred_dates, pred_values, true_values

def visualize_mode1(prediction_csv_path, train_data_path,
                    input_len=96, output_path="prediction_mode1.png"):
    """Visualize mode 1 predictions (future forecasting)."""
    try:
        scaler = load_and_fit_scaler(train_data_path)

        print(f"Loading prediction results: {prediction_csv_path}")
        pred_df = pd.read_csv(prediction_csv_path)
        input_dates, input_values, pred_dates, pred_values = _prepare_mode1_series(
            pred_df, scaler, input_len
        )

        plt.figure(figsize=(15, 8))
        plt.plot(
            input_dates,
            input_values,
            'o-',
            color='orange',
            linewidth=2,
            label=f'Input Data ({len(input_values)} steps)',
            alpha=0.8,
            markersize=4,
        )
        plt.plot(
            pred_dates,
            pred_values,
            's-',
            color='blue',
            linewidth=2,
            label=f'Predicted Output ({len(pred_values)} steps)',
            alpha=0.8,
            markersize=4,
        )

        if input_dates and pred_dates:
            _connect_points(
                plt,
                [input_dates, pred_dates],
                [input_values, pred_values],
                ['orange', 'blue'],
            )
            separation_time = input_dates[-1]
            plt.axvline(
                x=separation_time,
                color='black',
                linestyle='--',
                alpha=0.5,
                linewidth=1,
                label='Prediction Start',
            )

        plt.title('Mode 1: Future Prediction Visualization', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Load Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        _format_time_axis(plt.gca())
        plt.tight_layout()

        print(f"\nMode 1 Statistics:")
        print(f"Input - Mean: {np.mean(input_values):.2f}, Std: {np.std(input_values):.2f}")
        print(f"Prediction - Mean: {np.mean(pred_values):.2f}, Std: {np.std(pred_values):.2f}")

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mode 1 visualization saved to: {output_path}")
        plt.show()

    except Exception as exc:
        print(f"Error in visualize_mode1: {exc}")
        raise

def visualize_mode2(prediction_csv_path, train_data_path,
                    input_len=96, output_len=96, output_path="prediction_mode2.png"):
    """Visualize mode 2 predictions (historical sliding window)."""
    try:
        scaler = load_and_fit_scaler(train_data_path)

        print(f"Loading prediction results: {prediction_csv_path}")
        pred_df = pd.read_csv(prediction_csv_path)
        input_dates, input_values, pred_dates, pred_values, true_values = _prepare_mode2_series(
            pred_df, scaler
        )
        print(
            f"Mode 2: input={len(input_values)} pts, prediction={len(pred_values)} pts, true={len(true_values)} pts"
        )

        plt.figure(figsize=(15, 8))
        plt.plot(
            input_dates,
            input_values,
            'o-',
            color='orange',
            linewidth=2,
            label=f'Input Data ({input_len} steps)',
            alpha=0.8,
            markersize=4,
        )
        plt.plot(
            pred_dates,
            pred_values,
            's-',
            color='blue',
            linewidth=2,
            label=f'Predicted Output ({len(pred_values)} steps)',
            alpha=0.8,
            markersize=4,
        )
        plt.plot(
            pred_dates,
            true_values,
            '^-',
            color='green',
            linewidth=2,
            label=f'True Values ({len(true_values)} steps)',
            alpha=0.8,
            markersize=4,
        )

        if input_dates:
            _connect_points(
                plt,
                [input_dates, pred_dates, pred_dates],
                [input_values, pred_values, true_values],
                ['orange', 'blue', 'green'],
            )
            plt.axvline(
                x=input_dates[-1],
                color='black',
                linestyle='--',
                alpha=0.5,
                linewidth=1,
                label='Prediction Start',
            )

        plt.title('Mode 2: Historical Sliding Window Prediction Visualization', fontsize=16, fontweight='bold')
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Load Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        _format_time_axis(plt.gca())
        plt.tight_layout()

        print(f"\nMode 2 Statistics:")
        print(f"Input - Mean: {np.mean(input_values):.2f}, Std: {np.std(input_values):.2f}")
        print(f"Prediction - Mean: {np.mean(pred_values):.2f}, Std: {np.std(pred_values):.2f}")
        print(f"True Values - Mean: {np.mean(true_values):.2f}, Std: {np.std(true_values):.2f}")

        if len(pred_values) == len(true_values):
            mask = true_values != 0
            nonzero_true = true_values[mask]
            nonzero_pred = pred_values[mask]
            if len(nonzero_true) > 0:
                mae = np.mean(np.abs(pred_values - true_values))
                mse = np.mean((pred_values - true_values) ** 2)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((nonzero_true - nonzero_pred) / nonzero_true)) * 100

                print(f"\nPrediction Metrics:")
                print(f"MAE: {mae:.4f}")
                print(f"MSE: {mse:.4f}")
                print(f"RMSE: {rmse:.4f}")
                print(f"MAPE: {mape:.2f}%")

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Mode 2 visualization saved to: {output_path}")
        plt.show()

    except Exception as exc:
        print(f"Error in visualize_mode2: {exc}")
        raise

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

def main():
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

if __name__ == '__main__':
    main()
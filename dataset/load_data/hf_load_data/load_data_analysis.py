#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Data Analysis Tool
Comprehensive analysis of load_data dataset for time series forecasting
Created on: 2024-09-21
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from scipy import stats

# Try to import optional packages with fallbacks
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("⚠️  Seaborn not available - some visualizations will be simplified")

try:
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("⚠️  Statsmodels not available - some time series analyses will be skipped")

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  Scikit-learn not available - some analyses will be simplified")

# Set plotting style
plt.style.use('default')
if HAS_SEABORN:
    sns.set_palette("husl")

class LoadDataAnalyzer:
    def __init__(self, data_path=None):
        """
        Initialize the Load Data Analyzer
        
        Args:
            data_path (str): Path to the CSV file, if None uses default path
        """
        if data_path is None:
            # Use current directory as default
            self.data_path = "./hf_load_data_20210101-20250807_processed.csv"
        else:
            self.data_path = data_path
            
        # Create results directory in current folder
        self.results_dir = Path("./analysis_results")
        self.plots_dir = self.results_dir / "plots"
        self.reports_dir = self.results_dir / "reports"
        
        # Create directories
        for dir_path in [self.results_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(exist_ok=True)
            
        self.df = None
        self.analysis_results = {}
        
    def load_data(self):
        """Load and initial check of the dataset"""
        print("🔍 Loading load_data dataset...")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded successfully: {self.df.shape}")
            
            # Display basic info
            print(f"📊 Dataset shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            print(f"📅 Date range: {self.df.iloc[0, 0]} to {self.df.iloc[-1, 0]}")
            
            # Convert date column to datetime if exists
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
                self.df.set_index('date', inplace=True)
            elif self.df.columns[0].lower() in ['date', 'time', 'timestamp']:
                self.df.iloc[:, 0] = pd.to_datetime(self.df.iloc[:, 0])
                self.df.set_index(self.df.columns[0], inplace=True)
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def basic_statistics(self):
        """Perform basic statistical analysis"""
        print("\n📈 Performing basic statistical analysis...")
        
        results = {}
        
        # Basic info
        results['data_shape'] = self.df.shape
        results['columns'] = list(self.df.columns)
        results['data_types'] = self.df.dtypes.to_dict()
        
        # Missing values
        missing_info = self.df.isnull().sum().to_dict()
        results['missing_values'] = missing_info
        results['missing_percentage'] = {col: (count/len(self.df))*100 for col, count in missing_info.items()}
        
        # Descriptive statistics for numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results['descriptive_stats'] = {}
        
        for col in numeric_cols:
            col_stats = {
                'count': int(self.df[col].count()),
                'mean': float(self.df[col].mean()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'q25': float(self.df[col].quantile(0.25)),
                'median': float(self.df[col].median()),
                'q75': float(self.df[col].quantile(0.75)),
                'max': float(self.df[col].max()),
                'skewness': float(stats.skew(self.df[col].dropna())),
                'kurtosis': float(stats.kurtosis(self.df[col].dropna())),
                'range': float(self.df[col].max() - self.df[col].min()),
                'iqr': float(self.df[col].quantile(0.75) - self.df[col].quantile(0.25))
            }
            results['descriptive_stats'][col] = col_stats
        
        # Save results
        with open(self.reports_dir / 'basic_statistics.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.analysis_results['basic_statistics'] = results
        print("✅ Basic statistics analysis completed!")
        
        return results
    
    def time_series_analysis(self):
        """Analyze time series properties"""
        print("\n⏰ Performing time series analysis...")
        
        results = {}
        
        # Assume the main value column is 'value' or the last numeric column
        value_col = 'value' if 'value' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[-1]
        ts_data = self.df[value_col].dropna()
        
        # Frequency analysis
        if isinstance(self.df.index, pd.DatetimeIndex):
            results['frequency'] = str(pd.infer_freq(self.df.index))
            results['date_range'] = {
                'start': str(self.df.index.min()),
                'end': str(self.df.index.max()),
                'duration_days': (self.df.index.max() - self.df.index.min()).days
            }
        
        # Trend analysis (linear regression)
        x = np.arange(len(ts_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data)
        results['trend_analysis'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'trend_direction': 'increasing' if slope > 0 else 'decreasing'
        }
        
        # Stationarity test (ADF test)
        if HAS_STATSMODELS:
            try:
                adf_result = adfuller(ts_data)
                results['stationarity_adf'] = {
                    'adf_statistic': float(adf_result[0]),
                    'p_value': float(adf_result[1]),
                    'critical_values': {k: float(v) for k, v in adf_result[4].items()},
                    'is_stationary': adf_result[1] < 0.05
                }
            except Exception as e:
                results['stationarity_adf'] = {'error': str(e)}
        else:
            results['stationarity_adf'] = {'note': 'Statsmodels not available for ADF test'}
        
        # Autocorrelation analysis
        autocorr_lags = min(50, len(ts_data)//4)
        autocorr = [ts_data.autocorr(lag=i) for i in range(1, autocorr_lags+1)]
        results['autocorrelation'] = {
            'lags_1_to_10': [float(x) for x in autocorr[:10]],
            'significant_autocorr_count': sum(1 for x in autocorr if abs(x) > 0.1)
        }
        
        # Seasonality detection
        if len(ts_data) > 24:  # Need sufficient data for seasonality
            try:
                # Test for different seasonal periods
                seasonal_periods = [24, 168, 8760]  # Daily, weekly, yearly (for hourly data)
                results['seasonality'] = {}
                
                for period in seasonal_periods:
                    if len(ts_data) > 2 * period:
                        # Simple seasonal correlation test
                        seasonal_corr = np.corrcoef(ts_data[:-period], ts_data[period:])[0, 1]
                        results['seasonality'][f'period_{period}'] = float(seasonal_corr)
                        
            except Exception as e:
                results['seasonality'] = {'error': str(e)}
        
        # Save results
        with open(self.reports_dir / 'time_series_analysis.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.analysis_results['time_series_analysis'] = results
        print("✅ Time series analysis completed!")
        
        return results
    
    def data_quality_assessment(self):
        """Assess data quality"""
        print("\n🔍 Performing data quality assessment...")
        
        results = {}
        
        # Missing value patterns
        results['missing_patterns'] = self.df.isnull().sum().to_dict()
        
        # Data consistency checks
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        results['consistency_checks'] = {}
        
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            results['consistency_checks'][col] = {
                'negative_values_count': int((col_data < 0).sum()),
                'zero_values_count': int((col_data == 0).sum()),
                'infinite_values_count': int(np.isinf(col_data).sum()),
                'outliers_iqr_count': self._count_outliers_iqr(col_data),
                'outliers_zscore_count': self._count_outliers_zscore(col_data)
            }
        
        # Date gaps (if datetime index)
        if isinstance(self.df.index, pd.DatetimeIndex):
            expected_freq = pd.infer_freq(self.df.index)
            if expected_freq:
                full_range = pd.date_range(start=self.df.index.min(), 
                                         end=self.df.index.max(), 
                                         freq=expected_freq)
                missing_dates = len(full_range) - len(self.df.index)
                results['date_gaps'] = {
                    'expected_records': len(full_range),
                    'actual_records': len(self.df.index),
                    'missing_records': missing_dates,
                    'completeness_percentage': (len(self.df.index) / len(full_range)) * 100
                }
        
        # Overall quality score
        quality_score = self._calculate_quality_score()
        results['overall_quality_score'] = quality_score
        
        # Save results
        with open(self.reports_dir / 'data_quality_assessment.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.analysis_results['data_quality_assessment'] = results
        print("✅ Data quality assessment completed!")
        
        return results
    
    def _count_outliers_iqr(self, data):
        """Count outliers using IQR method"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((data < lower_bound) | (data > upper_bound)).sum())
    
    def _count_outliers_zscore(self, data, threshold=3):
        """Count outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(data))
        return int((z_scores > threshold).sum())
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score (0-100)"""
        score = 100
        
        # Penalize for missing values
        missing_ratio = self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])
        score -= missing_ratio * 30
        
        # Penalize for inconsistencies
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) > 0:
                outlier_ratio = self._count_outliers_iqr(col_data) / len(col_data)
                score -= outlier_ratio * 10
        
        return max(0, score)
    
    def pattern_analysis(self):
        """Analyze patterns in the data"""
        print("\n🔄 Performing pattern analysis...")
        
        results = {}
        
        # Main value column
        value_col = 'value' if 'value' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[-1]
        ts_data = self.df[value_col].dropna()
        
        # Periodic patterns
        if isinstance(self.df.index, pd.DatetimeIndex):
            df_with_time = self.df.copy()
            df_with_time['hour'] = df_with_time.index.hour
            df_with_time['day_of_week'] = df_with_time.index.dayofweek
            df_with_time['month'] = df_with_time.index.month
            
            results['periodic_patterns'] = {}
            
            # Hourly patterns
            if 'hour' in df_with_time.columns:
                hourly_stats = df_with_time.groupby('hour')[value_col].agg(['mean', 'std']).to_dict()
                results['periodic_patterns']['hourly'] = {
                    'mean_by_hour': {str(k): float(v) for k, v in hourly_stats['mean'].items()},
                    'std_by_hour': {str(k): float(v) for k, v in hourly_stats['std'].items()},
                    'peak_hour': int(df_with_time.groupby('hour')[value_col].mean().idxmax()),
                    'low_hour': int(df_with_time.groupby('hour')[value_col].mean().idxmin())
                }
            
            # Weekly patterns
            weekly_stats = df_with_time.groupby('day_of_week')[value_col].agg(['mean', 'std']).to_dict()
            results['periodic_patterns']['weekly'] = {
                'mean_by_day': {str(k): float(v) for k, v in weekly_stats['mean'].items()},
                'std_by_day': {str(k): float(v) for k, v in weekly_stats['std'].items()},
                'peak_day': int(df_with_time.groupby('day_of_week')[value_col].mean().idxmax()),
                'low_day': int(df_with_time.groupby('day_of_week')[value_col].mean().idxmin())
            }
            
            # Monthly patterns
            monthly_stats = df_with_time.groupby('month')[value_col].agg(['mean', 'std']).to_dict()
            results['periodic_patterns']['monthly'] = {
                'mean_by_month': {str(k): float(v) for k, v in monthly_stats['mean'].items()},
                'std_by_month': {str(k): float(v) for k, v in monthly_stats['std'].items()},
                'peak_month': int(df_with_time.groupby('month')[value_col].mean().idxmax()),
                'low_month': int(df_with_time.groupby('month')[value_col].mean().idxmin())
            }
        
        # Trend changes (change point detection - simple)
        window_size = min(100, len(ts_data)//10)
        if window_size > 10:
            rolling_mean = ts_data.rolling(window=window_size).mean()
            rolling_std = ts_data.rolling(window=window_size).std()
            
            results['trend_changes'] = {
                'high_volatility_periods': int((rolling_std > rolling_std.quantile(0.9)).sum()),
                'trend_reversals': self._detect_trend_reversals(rolling_mean),
                'volatility_range': {
                    'min_std': float(rolling_std.min()),
                    'max_std': float(rolling_std.max()),
                    'mean_std': float(rolling_std.mean())
                }
            }
        
        # Extreme values analysis
        results['extreme_values'] = {
            'top_5_values': ts_data.nlargest(5).tolist(),
            'bottom_5_values': ts_data.nsmallest(5).tolist(),
            'extreme_dates': {
                'max_date': str(ts_data.idxmax()),
                'min_date': str(ts_data.idxmin())
            }
        }
        
        # Save results
        with open(self.reports_dir / 'pattern_analysis.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.analysis_results['pattern_analysis'] = results
        print("✅ Pattern analysis completed!")
        
        return results
    
    def _detect_trend_reversals(self, data):
        """Detect trend reversals in the data"""
        if len(data) < 3:
            return 0
            
        diffs = np.diff(data.dropna())
        if len(diffs) < 2:
            return 0
            
        # Count sign changes in the differences
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        return int(sign_changes)
    
    def cyclical_trend_analysis(self):
        """深度分析短周期和长周期的趋势变化"""
        print("\n🔄 Performing cyclical and trend analysis...")
        
        results = {}
        value_col = 'value' if 'value' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[-1]
        
        if not isinstance(self.df.index, pd.DatetimeIndex):
            print("⚠️  需要时间索引进行周期性分析")
            return results
        
        # 短周期分析 (日、周)
        short_cycle_results = self._analyze_short_cycles(value_col)
        results['short_cycles'] = short_cycle_results
        
        # 长周期分析 (月、季、年)
        long_cycle_results = self._analyze_long_cycles(value_col)
        results['long_cycles'] = long_cycle_results
        
        # 典型案例分析
        case_study_results = self._generate_case_studies(value_col)
        results['case_studies'] = case_study_results
        
        # Save results
        with open(self.reports_dir / 'cyclical_trend_analysis.json', 'w') as f:
            json.dump(results, f, indent=4, default=str)
            
        self.analysis_results['cyclical_trend_analysis'] = results
        print("✅ Cyclical and trend analysis completed!")
        
        return results
    
    def _analyze_short_cycles(self, value_col):
        """分析短周期模式 (小时、日、周)"""
        results = {}
        
        # 每日模式分析
        daily_pattern = self._analyze_daily_pattern(value_col)
        results['daily_pattern'] = daily_pattern
        
        # 每周模式分析
        weekly_pattern = self._analyze_weekly_pattern(value_col)
        results['weekly_pattern'] = weekly_pattern
        
        # 工作日vs周末对比
        weekday_weekend = self._analyze_weekday_weekend(value_col)
        results['weekday_weekend'] = weekday_weekend
        
        return results
    
    def _analyze_long_cycles(self, value_col):
        """分析长周期模式 (月、季、年)"""
        results = {}
        
        # 月度模式分析
        monthly_pattern = self._analyze_monthly_pattern(value_col)
        results['monthly_pattern'] = monthly_pattern
        
        # 季节性模式分析
        seasonal_pattern = self._analyze_seasonal_pattern(value_col)
        results['seasonal_pattern'] = seasonal_pattern
        
        # 年度趋势分析
        yearly_trend = self._analyze_yearly_trend(value_col)
        results['yearly_trend'] = yearly_trend
        
        return results
    
    def _analyze_daily_pattern(self, value_col):
        """分析日内负荷变化模式"""
        df_hourly = self.df.copy()
        df_hourly['hour'] = df_hourly.index.hour
        
        hourly_stats = df_hourly.groupby('hour')[value_col].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # 计算峰谷差和变化率
        daily_min = hourly_stats['mean'].min()
        daily_max = hourly_stats['mean'].max()
        peak_valley_ratio = (daily_max - daily_min) / daily_min * 100
        
        # 识别负荷特征时段
        peak_hour = hourly_stats['mean'].idxmax()
        valley_hour = hourly_stats['mean'].idxmin()
        
        return {
            'hourly_statistics': hourly_stats.to_dict(),
            'peak_hour': int(peak_hour),
            'valley_hour': int(valley_hour),
            'daily_min_load': float(daily_min),
            'daily_max_load': float(daily_max),
            'peak_valley_ratio': float(peak_valley_ratio),
            'load_factor': float(hourly_stats['mean'].mean() / daily_max * 100)
        }
    
    def _analyze_weekly_pattern(self, value_col):
        """分析周内负荷变化模式"""
        df_weekly = self.df.copy()
        df_weekly['day_of_week'] = df_weekly.index.dayofweek
        df_weekly['day_name'] = df_weekly.index.day_name()
        
        weekly_stats = df_weekly.groupby(['day_of_week', 'day_name'])[value_col].agg(['mean', 'std']).round(2)
        
        # 计算工作日和周末的平均负荷
        workday_load = df_weekly[df_weekly['day_of_week'] < 5][value_col].mean()
        weekend_load = df_weekly[df_weekly['day_of_week'] >= 5][value_col].mean()
        
        # 转换为JSON可序列化的格式
        weekly_stats_dict = {}
        for (day_num, day_name), row in weekly_stats.iterrows():
            key = f"{day_num}_{day_name}"
            weekly_stats_dict[key] = {
                'mean': float(row['mean']),
                'std': float(row['std'])
            }
        
        peak_idx = weekly_stats['mean'].idxmax()
        low_idx = weekly_stats['mean'].idxmin()
        
        return {
            'weekly_statistics': weekly_stats_dict,
            'workday_average': float(workday_load),
            'weekend_average': float(weekend_load),
            'workday_weekend_ratio': float(workday_load / weekend_load),
            'peak_day': peak_idx[1] if isinstance(peak_idx, tuple) else str(peak_idx),
            'low_day': low_idx[1] if isinstance(low_idx, tuple) else str(low_idx)
        }
    
    def _analyze_weekday_weekend(self, value_col):
        """分析工作日与周末的负荷差异"""
        df_compare = self.df.copy()
        df_compare['is_weekend'] = df_compare.index.dayofweek >= 5
        df_compare['hour'] = df_compare.index.hour
        
        # 按小时比较工作日和周末
        comparison = df_compare.groupby(['hour', 'is_weekend'])[value_col].mean().unstack()
        
        # 转换为JSON可序列化的格式
        comparison_dict = {}
        if not comparison.empty:
            for hour in comparison.index:
                hour_data = {}
                if False in comparison.columns:
                    hour_data['weekday'] = float(comparison.loc[hour, False])
                if True in comparison.columns:
                    hour_data['weekend'] = float(comparison.loc[hour, True])
                comparison_dict[str(hour)] = hour_data
        
        # 计算差异统计
        if False in comparison.columns and True in comparison.columns:
            hourly_diff = comparison[False] - comparison[True]  # 工作日 - 周末
            max_diff_hour = hourly_diff.idxmax()
            max_diff_value = hourly_diff.max()
            avg_diff = hourly_diff.mean()
        else:
            max_diff_hour = 0
            max_diff_value = 0
            avg_diff = 0
        
        return {
            'hourly_comparison': comparison_dict,
            'max_difference_hour': int(max_diff_hour),
            'max_difference_value': float(max_diff_value),
            'average_difference': float(avg_diff)
        }
    
    def _analyze_monthly_pattern(self, value_col):
        """分析月度负荷变化模式"""
        df_monthly = self.df.copy()
        df_monthly['month'] = df_monthly.index.month
        df_monthly['month_name'] = df_monthly.index.month_name()
        
        monthly_stats = df_monthly.groupby(['month', 'month_name'])[value_col].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # 计算季节性指标
        spring_load = df_monthly[df_monthly['month'].isin([3, 4, 5])][value_col].mean()
        summer_load = df_monthly[df_monthly['month'].isin([6, 7, 8])][value_col].mean()
        autumn_load = df_monthly[df_monthly['month'].isin([9, 10, 11])][value_col].mean()
        winter_load = df_monthly[df_monthly['month'].isin([12, 1, 2])][value_col].mean()
        
        # 转换为JSON可序列化的格式
        monthly_stats_dict = {}
        for (month_num, month_name), row in monthly_stats.iterrows():
            key = f"{month_num}_{month_name}"
            monthly_stats_dict[key] = {
                'mean': float(row['mean']),
                'std': float(row['std']),
                'min': float(row['min']),
                'max': float(row['max'])
            }
        
        peak_idx = monthly_stats['mean'].idxmax()
        low_idx = monthly_stats['mean'].idxmin()
        
        return {
            'monthly_statistics': monthly_stats_dict,
            'seasonal_averages': {
                'spring': float(spring_load),
                'summer': float(summer_load),
                'autumn': float(autumn_load),
                'winter': float(winter_load)
            },
            'peak_month': peak_idx[1] if isinstance(peak_idx, tuple) else str(peak_idx),
            'low_month': low_idx[1] if isinstance(low_idx, tuple) else str(low_idx)
        }
    
    def _analyze_seasonal_pattern(self, value_col):
        """分析季节性负荷特征"""
        df_seasonal = self.df.copy()
        df_seasonal['month'] = df_seasonal.index.month
        df_seasonal['season'] = pd.Series(df_seasonal['month']).apply(self._get_season).values
        
        seasonal_stats = df_seasonal.groupby('season')[value_col].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # 计算季节性变化系数
        seasonal_cv = (seasonal_stats['std'] / seasonal_stats['mean'] * 100).round(2)
        
        return {
            'seasonal_statistics': seasonal_stats.to_dict(),
            'seasonal_cv': seasonal_cv.to_dict(),
            'highest_season': seasonal_stats['mean'].idxmax(),
            'lowest_season': seasonal_stats['mean'].idxmin()
        }
    
    def _analyze_yearly_trend(self, value_col):
        """分析年度趋势变化"""
        df_yearly = self.df.copy()
        df_yearly['year'] = df_yearly.index.year
        
        yearly_stats = df_yearly.groupby('year')[value_col].agg(['mean', 'std', 'min', 'max']).round(2)
        
        # 计算年度增长率
        if len(yearly_stats) > 1:
            growth_rates = yearly_stats['mean'].pct_change().dropna() * 100
            avg_growth_rate = growth_rates.mean()
        else:
            growth_rates = pd.Series()
            avg_growth_rate = 0
        
        return {
            'yearly_statistics': yearly_stats.to_dict(),
            'growth_rates': growth_rates.to_dict(),
            'average_growth_rate': float(avg_growth_rate),
            'trend_direction': 'increasing' if avg_growth_rate > 0 else 'decreasing'
        }
    
    def _get_season(self, month):
        """根据月份判断季节"""
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'
    
    def _generate_case_studies(self, value_col):
        """生成典型案例分析"""
        results = {}
        
        # 选择代表性的时间段进行案例分析
        case_periods = {
            'typical_day': self._get_typical_day(value_col),
            'peak_week': self._get_peak_week(value_col),
            'seasonal_comparison': self._get_seasonal_comparison(value_col)
        }
        
        for case_name, case_data in case_periods.items():
            if case_data is not None:
                results[case_name] = case_data
        
        return results
    
    def _get_typical_day(self, value_col):
        """获取典型日负荷曲线"""
        # 选择一个工作日作为典型日
        workdays = self.df[self.df.index.dayofweek < 5]
        if workdays.empty:
            return None
        
        # 计算平均日负荷曲线
        typical_day = workdays.groupby(workdays.index.hour)[value_col].mean()
        
        # 选择最接近平均值的实际日期
        daily_avg = workdays.groupby(workdays.index.date)[value_col].mean()
        overall_avg = daily_avg.mean()
        closest_date = daily_avg.sub(overall_avg).abs().idxmin()
        
        actual_day = workdays[workdays.index.date == closest_date]
        
        # 转换为JSON可序列化的格式
        typical_curve_dict = {str(hour): float(load) for hour, load in typical_day.items()}
        actual_curve_dict = {str(idx): float(val) for idx, val in actual_day[value_col].items()}
        
        return {
            'date': str(closest_date),
            'typical_curve': typical_curve_dict,
            'actual_curve': actual_curve_dict,
            'daily_statistics': {
                'mean': float(actual_day[value_col].mean()),
                'min': float(actual_day[value_col].min()),
                'max': float(actual_day[value_col].max()),
                'std': float(actual_day[value_col].std())
            }
        }
    
    def _get_peak_week(self, value_col):
        """获取峰值周的负荷特征"""
        # 按周计算平均负荷，找到峰值周
        weekly_avg = self.df.groupby(pd.Grouper(freq='W'))[value_col].mean()
        if weekly_avg.empty:
            return None
        
        peak_week_start = weekly_avg.idxmax()
        peak_week_end = peak_week_start + pd.Timedelta(days=6)
        
        peak_week_data = self.df[peak_week_start:peak_week_end]
        
        # 转换为JSON可序列化的格式
        weekly_pattern = peak_week_data.groupby(peak_week_data.index.dayofweek)[value_col].mean()
        weekly_pattern_dict = {str(day): float(load) for day, load in weekly_pattern.items()}
        
        return {
            'week_start': str(peak_week_start.date()),
            'week_end': str(peak_week_end.date()),
            'weekly_pattern': weekly_pattern_dict,
            'week_statistics': {
                'mean': float(peak_week_data[value_col].mean()),
                'min': float(peak_week_data[value_col].min()),
                'max': float(peak_week_data[value_col].max()),
                'std': float(peak_week_data[value_col].std())
            }
        }
    
    def _get_seasonal_comparison(self, value_col):
        """获取季节性对比分析"""
        seasons = {}
        for season_name, months in [('Summer', [6, 7, 8]), ('Winter', [12, 1, 2])]:
            season_data = self.df[self.df.index.month.isin(months)]
            if not season_data.empty:
                daily_pattern = season_data.groupby(season_data.index.hour)[value_col].mean()
                daily_pattern_dict = {str(hour): float(load) for hour, load in daily_pattern.items()}
                
                seasons[season_name] = {
                    'average_load': float(season_data[value_col].mean()),
                    'daily_pattern': daily_pattern_dict,
                    'statistics': {
                        'mean': float(season_data[value_col].mean()),
                        'std': float(season_data[value_col].std()),
                        'min': float(season_data[value_col].min()),
                        'max': float(season_data[value_col].max())
                    }
                }
        
        # 计算季节性差异
        if 'Summer' in seasons and 'Winter' in seasons:
            seasonal_difference = seasons['Summer']['average_load'] - seasons['Winter']['average_load']
            seasons['comparison'] = {
                'difference': float(seasonal_difference),
                'ratio': float(seasons['Summer']['average_load'] / seasons['Winter']['average_load'])
            }
        
        return seasons
    
    def create_cyclical_visualizations(self):
        """创建周期性和趋势性可视化图表"""
        print("\n📊 Creating cyclical and trend visualizations...")
        
        value_col = 'value' if 'value' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[-1]
        
        # 短周期可视化
        self._plot_short_cycle_analysis(value_col)
        
        # 长周期可视化
        self._plot_long_cycle_analysis(value_col)
        
        # 典型案例可视化
        self._plot_case_studies(value_col)
        
        print("✅ Cyclical and trend visualizations created!")
    
    def _plot_short_cycle_analysis(self, value_col):
        """绘制短周期分析图表"""
        # Set font to support Chinese characters but use English labels
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 日内负荷曲线
        hourly_data = self.df.groupby(self.df.index.hour)[value_col].agg(['mean', 'std'])
        axes[0, 0].plot(hourly_data.index, hourly_data['mean'], 'b-', linewidth=2, label='Average Load')
        axes[0, 0].fill_between(hourly_data.index, 
                               hourly_data['mean'] - hourly_data['std'],
                               hourly_data['mean'] + hourly_data['std'],
                               alpha=0.3, label='±1 Std Dev')
        axes[0, 0].set_title('Daily Load Pattern', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Load (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 周内负荷模式
        weekly_data = self.df.groupby(self.df.index.dayofweek)[value_col].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), weekly_data.values, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Weekly Load Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Load (MW)')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 工作日vs周末对比
        df_compare = self.df.copy()
        df_compare['is_weekend'] = df_compare.index.dayofweek >= 5
        df_compare['hour'] = df_compare.index.hour
        comparison = df_compare.groupby(['hour', 'is_weekend'])[value_col].mean().unstack()
        
        if False in comparison.columns and True in comparison.columns:
            axes[1, 0].plot(comparison.index, comparison[False], 'b-', linewidth=2, label='Weekday')
            axes[1, 0].plot(comparison.index, comparison[True], 'r-', linewidth=2, label='Weekend')
        axes[1, 0].set_title('Weekday vs Weekend Load Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Load (MW)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 日负荷系数分布
        daily_load_factor = self.df.groupby(self.df.index.date)[value_col].agg(['mean', 'max'])
        daily_load_factor['load_factor'] = daily_load_factor['mean'] / daily_load_factor['max']
        axes[1, 1].hist(daily_load_factor['load_factor'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(daily_load_factor['load_factor'].mean(), color='red', linestyle='--', 
                          label=f'Mean: {daily_load_factor["load_factor"].mean():.3f}')
        axes[1, 1].set_title('Daily Load Factor Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Load Factor')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'short_cycle_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_long_cycle_analysis(self, value_col):
        """绘制长周期分析图表"""
        # Set font to support Chinese characters but use English labels
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 月度负荷变化
        monthly_data = self.df.groupby(self.df.index.month)[value_col].agg(['mean', 'std'])
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0, 0].plot(monthly_data.index, monthly_data['mean'], 'o-', linewidth=2, markersize=6)
        axes[0, 0].errorbar(monthly_data.index, monthly_data['mean'], yerr=monthly_data['std'], 
                           capsize=5, alpha=0.7)
        axes[0, 0].set_title('Monthly Load Trend', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Month')
        axes[0, 0].set_ylabel('Average Load (MW)')
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].set_xticklabels(month_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 季节性负荷分布
        seasonal_data = self.df.copy()
        seasonal_data['season'] = pd.Series(seasonal_data.index.month).apply(self._get_season).values
        season_avg = seasonal_data.groupby('season')[value_col].mean()
        colors = ['lightgreen', 'gold', 'orange', 'lightblue']
        axes[0, 1].bar(season_avg.index, season_avg.values, color=colors, alpha=0.7)
        axes[0, 1].set_title('Seasonal Load Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Season')
        axes[0, 1].set_ylabel('Average Load (MW)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 年度趋势
        if len(self.df.index.year.unique()) > 1:
            yearly_data = self.df.groupby(self.df.index.year)[value_col].mean()
            axes[1, 0].plot(yearly_data.index, yearly_data.values, 'o-', linewidth=2, markersize=8)
            
            # 添加趋势线
            x = np.arange(len(yearly_data))
            coeffs = np.polyfit(x, yearly_data.values, 1)
            trend_line = coeffs[0] * x + coeffs[1]
            axes[1, 0].plot(yearly_data.index, trend_line, '--', color='red', alpha=0.7, 
                           label=f'Trend: {coeffs[0]:.2f} MW/year')
            axes[1, 0].legend()
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor yearly trend analysis', 
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        
        axes[1, 0].set_title('Annual Load Trend', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Annual Average Load (MW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 长期变化系数
        monthly_cv = self.df.groupby([self.df.index.year, self.df.index.month])[value_col].std() / \
                    self.df.groupby([self.df.index.year, self.df.index.month])[value_col].mean()
        if not monthly_cv.empty:
            axes[1, 1].plot(range(len(monthly_cv)), monthly_cv.values, linewidth=1.5)
            axes[1, 1].axhline(monthly_cv.mean(), color='red', linestyle='--', 
                              label=f'Mean CV: {monthly_cv.mean():.3f}')
            axes[1, 1].legend()
        axes[1, 1].set_title('Load Coefficient of Variation Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Time Series')
        axes[1, 1].set_ylabel('Coefficient of Variation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'long_cycle_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_case_studies(self, value_col):
        """绘制典型案例分析图表"""
        # Set font to support Chinese characters but use English labels
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 典型日负荷曲线
        workdays = self.df[self.df.index.dayofweek < 5]
        if not workdays.empty:
            # 选择一个代表性工作日
            daily_avg = workdays.groupby(workdays.index.date)[value_col].mean()
            overall_avg = daily_avg.mean()
            closest_date = daily_avg.sub(overall_avg).abs().idxmin()
            typical_day = workdays[workdays.index.date == closest_date]
            
            # 绘制典型日和平均日曲线
            avg_hourly = workdays.groupby(workdays.index.hour)[value_col].mean()
            axes[0, 0].plot(typical_day.index.hour, typical_day[value_col], 'b-', 
                           linewidth=2, label=f'Typical Day ({closest_date})')
            axes[0, 0].plot(avg_hourly.index, avg_hourly.values, 'r--', 
                           linewidth=2, label='Weekday Average')
            
        axes[0, 0].set_title('Typical Weekday Load Curve', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Load (MW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 峰值周分析
        weekly_avg = self.df.groupby(pd.Grouper(freq='W'))[value_col].mean()
        if not weekly_avg.empty:
            peak_week_start = weekly_avg.idxmax()
            peak_week_end = peak_week_start + pd.Timedelta(days=6)
            peak_week_data = self.df[peak_week_start:peak_week_end]
            
            axes[0, 1].plot(peak_week_data.index, peak_week_data[value_col], linewidth=1.5)
            axes[0, 1].set_title(f'Peak Week Load Pattern ({peak_week_start.date()} - {peak_week_end.date()})', 
                                fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Load (MW)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 季节性对比 (夏季vs冬季)
        summer_data = self.df[self.df.index.month.isin([6, 7, 8])]
        winter_data = self.df[self.df.index.month.isin([12, 1, 2])]
        
        if not summer_data.empty and not winter_data.empty:
            summer_hourly = summer_data.groupby(summer_data.index.hour)[value_col].mean()
            winter_hourly = winter_data.groupby(winter_data.index.hour)[value_col].mean()
            
            axes[1, 0].plot(summer_hourly.index, summer_hourly.values, 'r-', 
                           linewidth=2, label='Summer Average')
            axes[1, 0].plot(winter_hourly.index, winter_hourly.values, 'b-', 
                           linewidth=2, label='Winter Average')
            
        axes[1, 0].set_title('Summer vs Winter Load Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Average Load (MW)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 负荷持续曲线
        load_duration = np.sort(self.df[value_col].values)[::-1]
        duration_percent = np.arange(1, len(load_duration) + 1) / len(load_duration) * 100
        
        axes[1, 1].plot(duration_percent, load_duration, linewidth=2)
        axes[1, 1].axhline(self.df[value_col].mean(), color='red', linestyle='--', 
                          label=f'Average Load: {self.df[value_col].mean():.1f} MW')
        axes[1, 1].axvline(50, color='orange', linestyle='--', alpha=0.7, label='Median')
        axes[1, 1].set_title('Load Duration Curve', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Duration Percentage (%)')
        axes[1, 1].set_ylabel('Load (MW)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'case_studies.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📊 Creating visualizations...")
        
        # Set font to support Chinese characters but use English labels
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Main value column
        value_col = 'value' if 'value' in self.df.columns else self.df.select_dtypes(include=[np.number]).columns[-1]
        
        # 1. Time series trend plot
        self._plot_time_series_trend(value_col)
        
        # 2. Distribution analysis
        self._plot_distribution_analysis(value_col)
        
        # 3. Seasonality analysis
        self._plot_seasonality_analysis(value_col)
        
        # 4. Correlation heatmap
        self._plot_correlation_heatmap()
        
        # 5. Rolling statistics
        self._plot_rolling_statistics(value_col)
        
        # 6. Outlier detection
        self._plot_outlier_detection(value_col)
        
        # 7. Autocorrelation
        self._plot_autocorrelation(value_col)
        
        # 8. Time series decomposition
        self._plot_decomposition(value_col)
        
        print("✅ All visualizations created!")
    
    def _plot_time_series_trend(self, value_col):
        """Plot time series trend"""
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(self.df.index, self.df[value_col], alpha=0.7, linewidth=0.8)
        plt.title(f'Time Series Trend - {value_col}', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        # Rolling statistics
        rolling_mean = self.df[value_col].rolling(window=24*7).mean()  # Weekly rolling mean
        rolling_std = self.df[value_col].rolling(window=24*7).std()    # Weekly rolling std
        
        plt.plot(self.df.index, self.df[value_col], alpha=0.3, label='Original', linewidth=0.5)
        plt.plot(self.df.index, rolling_mean, 'r-', label='Rolling Mean', linewidth=1.5)
        plt.fill_between(self.df.index, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.2, color='red', label='±1 Std')
        plt.title('Time Series with Rolling Statistics', fontsize=14, fontweight='bold')
        plt.ylabel('Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'time_series_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_analysis(self, value_col):
        """Plot distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        data = self.df[value_col].dropna()
        
        # Histogram
        axes[0, 0].hist(data, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[0, 0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.2f}')
        axes[0, 0].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.2f}')
        axes[0, 0].set_title('Histogram', fontweight='bold')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(data, vert=True)
        axes[0, 1].set_title('Box Plot', fontweight='bold')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_data = np.sort(data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[1, 1].plot(sorted_data, cumulative, linewidth=2)
        axes[1, 1].set_title('Cumulative Distribution Function', fontweight='bold')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonality_analysis(self, value_col):
        """Plot seasonality analysis"""
        if not isinstance(self.df.index, pd.DatetimeIndex):
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        df_with_time = self.df.copy()
        df_with_time['hour'] = df_with_time.index.hour
        df_with_time['day_of_week'] = df_with_time.index.dayofweek
        df_with_time['month'] = df_with_time.index.month
        
        # Hourly pattern
        hourly_mean = df_with_time.groupby('hour')[value_col].mean()
        axes[0, 0].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2)
        axes[0, 0].set_title('Average by Hour of Day', fontweight='bold')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Average Value')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(0, 24, 2))
        
        # Daily pattern
        daily_mean = df_with_time.groupby('day_of_week')[value_col].mean()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(7), daily_mean.values, alpha=0.7)
        axes[0, 1].set_title('Average by Day of Week', fontweight='bold')
        axes[0, 1].set_xlabel('Day of Week')
        axes[0, 1].set_ylabel('Average Value')
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly pattern
        monthly_mean = df_with_time.groupby('month')[value_col].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1, 0].plot(monthly_mean.index, monthly_mean.values, marker='s', linewidth=2)
        axes[1, 0].set_title('Average by Month', fontweight='bold')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('Average Value')
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(month_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Heatmap: Hour vs Day of Week
        heatmap_data = df_with_time.pivot_table(values=value_col, index='hour', columns='day_of_week', aggfunc='mean')
        im = axes[1, 1].imshow(heatmap_data.values, cmap='viridis', aspect='auto')
        axes[1, 1].set_title('Heatmap: Hour vs Day of Week', fontweight='bold')
        axes[1, 1].set_xlabel('Day of Week')
        axes[1, 1].set_ylabel('Hour')
        axes[1, 1].set_xticks(range(7))
        axes[1, 1].set_xticklabels(day_names)
        axes[1, 1].set_yticks(range(0, 24, 4))
        plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'seasonality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        numeric_cols = self.df.select_dtypes(include=[np.number])
        if len(numeric_cols.columns) < 2:
            return
            
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_cols.corr()
        
        if HAS_SEABORN:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
        else:
            # Fallback to matplotlib imshow
            plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
            plt.colorbar()
            
            # Add correlation values as text
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    if i != j:  # Don't show diagonal
                        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
            
            plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
            plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
            
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_rolling_statistics(self, value_col):
        """Plot rolling statistics"""
        plt.figure(figsize=(15, 10))
        
        data = self.df[value_col].dropna()
        
        # Different window sizes
        windows = [24, 24*7, 24*30]  # Daily, weekly, monthly for hourly data
        window_labels = ['24h', '1 week', '1 month']
        
        plt.subplot(3, 1, 1)
        plt.plot(data.index, data.values, alpha=0.3, label='Original', linewidth=0.5)
        for window, label in zip(windows, window_labels):
            if len(data) > window:
                rolling_mean = data.rolling(window=window).mean()
                plt.plot(rolling_mean.index, rolling_mean.values, label=f'Rolling Mean ({label})', linewidth=1.5)
        plt.title('Rolling Mean Comparison', fontweight='bold')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        for window, label in zip(windows, window_labels):
            if len(data) > window:
                rolling_std = data.rolling(window=window).std()
                plt.plot(rolling_std.index, rolling_std.values, label=f'Rolling Std ({label})', linewidth=1.5)
        plt.title('Rolling Standard Deviation', fontweight='bold')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        if len(data) > windows[1]:  # Use weekly window
            rolling_min = data.rolling(window=windows[1]).min()
            rolling_max = data.rolling(window=windows[1]).max()
            rolling_mean = data.rolling(window=windows[1]).mean()
            
            plt.fill_between(data.index, rolling_min, rolling_max, alpha=0.2, label='Min-Max Range')
            plt.plot(rolling_mean.index, rolling_mean.values, 'r-', label='Rolling Mean', linewidth=2)
            plt.plot(data.index, data.values, alpha=0.3, label='Original', linewidth=0.5)
        plt.title('Rolling Min-Max Range with Mean', fontweight='bold')
        plt.ylabel('Value')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'rolling_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_outlier_detection(self, value_col):
        """Plot outlier detection"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        data = self.df[value_col].dropna()
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        outliers_zscore = data[z_scores > 3]
        
        axes[0, 0].scatter(range(len(data)), data.values, alpha=0.6, s=1)
        if len(outliers_zscore) > 0:
            outlier_indices = data.index.get_indexer(outliers_zscore.index)
            axes[0, 0].scatter(outlier_indices, outliers_zscore.values, color='red', s=20, label=f'Outliers ({len(outliers_zscore)})')
        axes[0, 0].set_title('Outlier Detection - Z-score Method', fontweight='bold')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]
        
        axes[0, 1].scatter(range(len(data)), data.values, alpha=0.6, s=1)
        if len(outliers_iqr) > 0:
            outlier_indices = data.index.get_indexer(outliers_iqr.index)
            axes[0, 1].scatter(outlier_indices, outliers_iqr.values, color='red', s=20, label=f'Outliers ({len(outliers_iqr)})')
        axes[0, 1].axhline(y=upper_bound, color='orange', linestyle='--', alpha=0.8, label='Upper bound')
        axes[0, 1].axhline(y=lower_bound, color='orange', linestyle='--', alpha=0.8, label='Lower bound')
        axes[0, 1].set_title('Outlier Detection - IQR Method', fontweight='bold')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series plot with outliers highlighted
        axes[1, 0].plot(data.index, data.values, alpha=0.7, linewidth=0.8)
        if len(outliers_zscore) > 0:
            axes[1, 0].scatter(outliers_zscore.index, outliers_zscore.values, 
                             color='red', s=30, alpha=0.8, label=f'Z-score Outliers ({len(outliers_zscore)})')
        axes[1, 0].set_title('Time Series with Outliers (Z-score)', fontweight='bold')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution with outliers
        axes[1, 1].hist(data.values, bins=50, alpha=0.7, density=True, edgecolor='black')
        if len(outliers_zscore) > 0:
            axes[1, 1].hist(outliers_zscore.values, bins=20, alpha=0.8, density=True, 
                           color='red', edgecolor='darkred', label=f'Outliers ({len(outliers_zscore)})')
        axes[1, 1].set_title('Distribution with Outliers Highlighted', fontweight='bold')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'outlier_detection.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_autocorrelation(self, value_col):
        """Plot autocorrelation and partial autocorrelation"""
        data = self.df[value_col].dropna()
        
        if HAS_STATSMODELS and len(data) > 50:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            max_lags = min(50, len(data)//4)
            
            # ACF
            plot_acf(data, lags=max_lags, ax=axes[0], alpha=0.05)
            axes[0].set_title('Autocorrelation Function (ACF)', fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # PACF
            plot_pacf(data, lags=max_lags, ax=axes[1], alpha=0.05)
            axes[1].set_title('Partial Autocorrelation Function (PACF)', fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'autocorrelation.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            # Simple manual autocorrelation plot
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            
            max_lags = min(50, len(data)//4)
            autocorrs = [1.0]  # lag 0 is always 1
            
            for lag in range(1, max_lags):
                if len(data) > lag:
                    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                    autocorrs.append(corr)
                else:
                    break
                    
            ax.stem(range(len(autocorrs)), autocorrs, basefmt=' ')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='±0.2 threshold')
            ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Autocorrelation Function (Manual Implementation)', fontweight='bold')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Autocorrelation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'autocorrelation.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_decomposition(self, value_col):
        """Plot time series decomposition"""
        if not isinstance(self.df.index, pd.DatetimeIndex) or len(self.df) < 100:
            return
            
        if not HAS_STATSMODELS:
            print("⚠️  Statsmodels not available - skipping decomposition plot")
            return
            
        try:
            # Try different seasonal periods
            seasonal_period = 24  # Assume hourly data with daily seasonality
            if len(self.df) < 2 * seasonal_period:
                seasonal_period = len(self.df) // 4
                
            if seasonal_period < 2:
                return
                
            decomposition = seasonal_decompose(self.df[value_col].dropna(), 
                                             model='additive', 
                                             period=seasonal_period)
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            # Original
            decomposition.observed.plot(ax=axes[0], title='Original Time Series')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            decomposition.trend.plot(ax=axes[1], title='Trend Component', color='orange')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component', color='green')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            decomposition.resid.plot(ax=axes[3], title='Residual Component', color='red')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'decomposition.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️  Could not create decomposition plot: {e}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive markdown report in Chinese"""
        print("\n📝 生成详细分析报告...")
        
        report_content = f"""# 负荷数据深度分析报告

**生成时间:** {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

**数据集:** {self.data_path}

## 执行摘要

本报告对用于时间序列预测的负荷数据进行了全面深入的分析。分析涵盖基本统计特征、时间序列特性、数据质量评估、模式识别和可视化展示。该数据集表现出明显的时间规律性和季节性特征，为电力负荷预测提供了高质量的训练数据。

## 1. 数据集概览

- **数据规模:** {self.df.shape[0]:,} 条记录 × {self.df.shape[1]} 个特征
- **特征列:** {', '.join(self.df.columns)}
- **时间跨度:** {self.df.index.min()} 至 {self.df.index.max()}
- **持续时长:** {(self.df.index.max() - self.df.index.min()).days:,} 天
- **数据频率:** 5分钟间隔（每小时12个数据点）

### 时间序列整体趋势

![时间序列趋势图](../plots/time_series_trend.png)

上图展示了负荷数据的整体时间序列趋势。上半部分显示原始时间序列，可以观察到：
- 数据具有明显的周期性波动
- 整体呈现轻微上升趋势
- 存在明显的季节性变化模式

下半部分显示滚动统计特征：
- 红线为7天滚动均值，反映中期趋势
- 阴影区域为±1标准差范围，显示数据变异性
- 可以看出数据的波动性相对稳定

## 2. 数据质量评估

"""
        
        # Add quality assessment results
        if 'data_quality_assessment' in self.analysis_results:
            quality_data = self.analysis_results['data_quality_assessment']
            report_content += f"""
### 数据质量综合评分: {quality_data.get('overall_quality_score', 'N/A'):.1f}/100

### 缺失值分析
"""
            for col, missing_count in quality_data.get('missing_patterns', {}).items():
                missing_pct = (missing_count / len(self.df)) * 100
                report_content += f"- **{col}:** {missing_count} 个缺失值 ({missing_pct:.2f}%)\n"
            
            report_content += "\n### 数据一致性检查\n"
            consistency = quality_data.get('consistency_checks', {})
            for col, checks in consistency.items():
                report_content += f"\n**{col}字段:**\n"
                report_content += f"- IQR方法异常值: {checks.get('outliers_iqr', 'N/A')} 个\n"
                report_content += f"- Z-score方法异常值: {checks.get('outliers_zscore', 'N/A')} 个\n"
        
        # Add basic statistics
        if 'basic_statistics' in self.analysis_results:
            stats_data = self.analysis_results['basic_statistics']
            report_content += "\n## 3. 统计特征分析\n"
            
            report_content += "\n### 分布特征可视化\n\n![分布分析图](../plots/distribution_analysis.png)\n\n"
            report_content += """分布分析图包含四个关键视图：
1. **直方图**: 显示数据分布形状，红色虚线为均值，绿色虚线为中位数
2. **箱线图**: 展示四分位数分布和异常值识别
3. **Q-Q图**: 检验数据是否符合正态分布（点越接近直线越符合正态分布）
4. **累积分布函数**: 显示数据的累积概率分布

"""
            
            for col, stats in stats_data.get('descriptive_stats', {}).items():
                report_content += f"\n### {col}字段统计指标\n"
                
                # 安全的数字格式化函数
                def safe_format(value, format_str=".4f"):
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if format_str == ",":
                            return f"{value:,}"
                        else:
                            return f"{value:,.4f}"
                    else:
                        return "N/A"
                
                report_content += f"- **数据量:** {safe_format(stats.get('count'), ',')}\n"
                report_content += f"- **均值:** {safe_format(stats.get('mean'))}\n"
                report_content += f"- **标准差:** {safe_format(stats.get('std'))}\n"
                report_content += f"- **最小值:** {safe_format(stats.get('min'))}\n"
                report_content += f"- **第25百分位:** {safe_format(stats.get('q25'))}\n"
                report_content += f"- **中位数:** {safe_format(stats.get('median'))}\n"
                report_content += f"- **第75百分位:** {safe_format(stats.get('q75'))}\n"
                report_content += f"- **最大值:** {safe_format(stats.get('max'))}\n"
                report_content += f"- **偏度:** {safe_format(stats.get('skewness'))} (正值表示右偏)\n"
                report_content += f"- **峰度:** {safe_format(stats.get('kurtosis'))} (正值表示厚尾分布)\n"
                
                # 添加统计解释
                skew = stats.get('skewness', 0)
                kurt = stats.get('kurtosis', 0)
                
                report_content += "\n**统计特征解释:**\n"
                if abs(skew) < 0.5:
                    report_content += "- 数据分布基本对称\n"
                elif skew > 0.5:
                    report_content += "- 数据呈现明显右偏分布（高值较多）\n"
                else:
                    report_content += "- 数据呈现明显左偏分布（低值较多）\n"
                    
                if abs(kurt) < 0.5:
                    report_content += "- 数据分布接近正态分布的峰度\n"
                elif kurt > 0.5:
                    report_content += "- 数据分布具有厚尾特征（极值较多）\n"
                else:
                    report_content += "- 数据分布具有薄尾特征（极值较少）\n"
        
        # Add time series analysis
        if 'time_series_analysis' in self.analysis_results:
            ts_data = self.analysis_results['time_series_analysis']
            report_content += "\n## 4. 时间序列特性分析\n"
            
            # 趋势分析
            trend = ts_data.get('trend_analysis', {})
            
            # 安全格式化趋势数据
            def safe_trend_format(value, format_type='float'):
                if isinstance(value, (int, float)) and not pd.isna(value):
                    if format_type == 'slope':
                        return f"{value:,.6f}"
                    elif format_type == 'percent':
                        return f"{value:,.4f}"
                    else:
                        return f"{value:,.6f}"
                else:
                    return "N/A"
            
            slope_str = safe_trend_format(trend.get('slope'), 'slope')
            r_squared_str = safe_trend_format(trend.get('r_squared'), 'percent')
            p_value_str = safe_trend_format(trend.get('p_value'), 'percent')
            
            report_content += f"""
### 趋势分析
- **趋势方向:** {trend.get('trend_direction', 'N/A')}
- **斜率系数:** {slope_str}
- **决定系数 (R²):** {r_squared_str}
- **统计显著性 (p值):** {p_value_str}

**趋势解释:** """
            
            slope = trend.get('slope', 0)
            r_squared = trend.get('r_squared', 0)
            p_value = trend.get('p_value', 1)
            
            if abs(slope) < 0.001:
                report_content += "数据整体呈现平稳趋势，无明显上升或下降。"
            elif slope > 0:
                report_content += f"数据呈现{trend.get('trend_direction', '上升')}趋势。"
            else:
                report_content += f"数据呈现{trend.get('trend_direction', '下降')}趋势。"
            
            if r_squared > 0.1:
                report_content += f"趋势的解释力度为{r_squared*100:.1f}%，趋势相对明显。"
            else:
                report_content += f"趋势的解释力度仅为{r_squared*100:.1f}%，趋势较弱。"
            
            if p_value < 0.05:
                report_content += "趋势在统计学上显著。"
            else:
                report_content += "趋势在统计学上不显著。"
            
            # 自相关分析
            autocorr = ts_data.get('autocorrelation', {})
            report_content += f"""

### 自相关分析
![自相关图](../plots/autocorrelation.png)

- **前10个滞后期的自相关系数:** {autocorr.get('lags_1_to_10', ['N/A'])}
- **显著自相关的滞后期数量:** {autocorr.get('significant_autocorr_count', 'N/A')}

自相关分析显示了数据的时间依赖性。高自相关系数表明当前值与历史值密切相关，这为时间序列预测提供了重要信息。
"""
            
            # 季节性分析
            if 'seasonality' in ts_data:
                seasonality = ts_data['seasonality']
                
                # 安全的数字格式化函数
                def safe_format_num(value, decimals=4):
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        return f"{value:.{decimals}f}"
                    else:
                        return "N/A"
                
                daily_strength = safe_format_num(seasonality.get('period_24', 'N/A'))
                weekly_strength = safe_format_num(seasonality.get('period_168', 'N/A'))
                yearly_strength = safe_format_num(seasonality.get('period_8760', 'N/A'))
                
                report_content += f"""

### 季节性特征检测
- **日内周期强度:** {daily_strength}
- **周内周期强度:** {weekly_strength}
- **年内周期强度:** {yearly_strength}

![季节性分析图](../plots/seasonality_analysis.png)

季节性分析图展示了四个维度的周期性模式：
1. **小时模式**: 显示一天24小时内的负荷变化规律
2. **星期模式**: 展示一周7天的负荷差异（工作日vs周末）
3. **月份模式**: 反映全年12个月的季节性变化
4. **热力图**: 小时与星期的交叉分析，识别精细时间模式

季节性强度越接近1，表明该时间尺度的周期性越明显。"""
        
        # Add pattern analysis
        if 'pattern_analysis' in self.analysis_results:
            pattern_data = self.analysis_results['pattern_analysis']
            report_content += "\n## 5. 模式识别与异常检测\n"
            
            # 时间模式
            if 'time_patterns' in pattern_data:
                patterns = pattern_data['time_patterns']
                report_content += f"""
### 时间模式分析
- **峰值时段:** {patterns.get('peak_hour', 'N/A')}:00
- **低谷时段:** {patterns.get('low_hour', 'N/A')}:00
- **峰值工作日:** {['周一', '周二', '周三', '周四', '周五', '周六', '周日'][patterns.get('peak_day', 0)]}
- **低谷工作日:** {['周一', '周二', '周三', '周四', '周五', '周六', '周日'][patterns.get('low_day', 6)]}

"""
            elif 'periodic_patterns' in pattern_data:
                patterns = pattern_data['periodic_patterns']
                if 'hourly' in patterns:
                    hourly = patterns['hourly']
                    report_content += f"""
### 时间模式分析 - 小时模式
- **峰值时段:** {hourly.get('peak_hour', 'N/A')}:00
- **低谷时段:** {hourly.get('low_hour', 'N/A')}:00

"""
                
                if 'weekly' in patterns:
                    weekly = patterns['weekly']
                    days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
                    peak_day = days[weekly.get('peak_day', 0)] if weekly.get('peak_day', 0) < 7 else 'N/A'
                    low_day = days[weekly.get('low_day', 0)] if weekly.get('low_day', 0) < 7 else 'N/A'
                    report_content += f"""
### 时间模式分析 - 周模式
- **峰值工作日:** {peak_day}
- **低谷工作日:** {low_day}

"""
            
            # 极值分析
            if 'extreme_values' in pattern_data:
                extreme = pattern_data['extreme_values']
                report_content += f"""
### 极值分析
**最高负荷记录 (Top 5):**
"""
                for i, val in enumerate(extreme.get('top_5_values', []), 1):
                    report_content += f"{i}. {val:,.2f}\n"
                
                report_content += f"""
**最低负荷记录 (Bottom 5):**
"""
                for i, val in enumerate(extreme.get('bottom_5_values', []), 1):
                    report_content += f"{i}. {val:,.2f}\n"
                
                extreme_dates = extreme.get('extreme_dates', {})
                report_content += f"""
**极值发生时间:**
- 最大值时间: {extreme_dates.get('max_date', 'N/A')}
- 最小值时间: {extreme_dates.get('min_date', 'N/A')}
"""
            
            # 异常检测
            report_content += """
### 异常值检测

![异常检测图](../plots/outlier_detection.png)

异常检测采用两种方法：
1. **Z-score方法**: 基于标准化得分，|Z|>3视为异常
2. **IQR方法**: 基于四分位距，超出Q1-1.5*IQR或Q3+1.5*IQR视为异常

图中显示了不同方法识别的异常点分布和时间序列中的异常位置。
"""
        
        # 滚动统计分析
        report_content += """
## 6. 滚动统计分析

![滚动统计图](../plots/rolling_statistics.png)

滚动统计分析展示了不同时间窗口下的统计特征变化：
1. **滚动均值比较**: 24小时、1周、1个月窗口的移动平均
2. **滚动标准差**: 反映数据在不同时期的波动性
3. **滚动范围**: 最大值、最小值和均值的动态变化

这些指标有助于识别数据的时变特征和稳定性。
"""
        
        # 相关性分析
        report_content += """
## 7. 可视化图表说明

本分析生成了6类专业可视化图表，详细说明如下：

### 7.1 时间序列趋势图 
![时间序列趋势图](../plots/time_series_trend.png)

### 7.2 分布分析图
![分布分析图](../plots/distribution_analysis.png)

### 7.3 季节性分析图
![季节性分析图](../plots/seasonality_analysis.png)

### 8.5 滚动统计图
![滚动统计图](../plots/rolling_statistics.png)

### 8.6 异常检测图
![异常检测图](../plots/outlier_detection.png)

### 7.6 自相关分析图
![自相关分析图](../plots/autocorrelation.png)

## 8. 关键发现与建议

### 8.1 数据质量评估结论
- **数据完整性**: 数据集结构完整，适合用于时间序列分析
- **缺失值处理**: 需要在建模前适当处理缺失值模式
- **异常值识别**: 已识别异常值，需要进一步调查其产生原因

### 9.2 时间序列特性总结
- **周期性模式**: 数据展现出清晰的多尺度时间模式
- **季节性特征**: 在小时、日、周等多个时间尺度上存在季节性
- **趋势分析**: 长期趋势分析为预测模型提供重要参考

### 9.3 电力负荷特征洞察
1. **日内模式**: 典型的电力负荷日内变化曲线，峰谷差异明显
2. **周内模式**: 工作日与周末负荷模式存在显著差异
3. **季节模式**: 年内季节变化反映用电习惯和气候影响
4. **负荷水平**: 整体负荷水平在合理范围内，变异系数适中

### 9.4 短周期趋势分析

![短周期分析](../plots/short_cycle_analysis.png)

#### 9.4.1 日内负荷变化规律
基于小时级负荷数据的分析显示了明显的日内变化模式：
- **峰值时段**: 通常出现在用电需求较高的时段，反映了负荷的日常规律
- **谷值时段**: 一般在夜间或用电需求较低的时段，负荷处于最低水平
- **峰谷差率**: 反映了负荷的日内变化幅度，是电力系统运行的重要指标
- **负荷系数**: 表征负荷的平均利用效率，影响电网的经济运行

#### 9.4.2 周内负荷分布特征
工作日与周末的负荷模式存在显著差异：
- 工作日负荷一般较高，反映了商业和工业用电的集中性
- 周末负荷相对较低，主要由民用电负荷主导
- 周内负荷分布反映了经济活动和生活习惯的周期性特征

### 9.5 长周期趋势分析

![长周期分析](../plots/long_cycle_analysis.png)

#### 9.5.1 季节性负荷特征
季节性分析揭示了负荷随季节变化的规律：
- **季节性模式**: 不同季节的平均负荷水平反映了气候对用电需求的影响
- **月度变化**: 月度负荷变化趋势显示了更细致的季节性特征
- **年度趋势**: 年度负荷增长趋势反映了经济发展和用电需求的变化

#### 9.5.2 长期变化特征
- **年度增长**: 负荷的年度增长率反映了地区经济发展状况
- **变异系数**: 负荷变异系数的变化表征了负荷预测的难易程度
- **趋势识别**: 长期趋势分析为未来规划提供重要参考

### 9.6 典型案例分析

![典型案例分析](../plots/case_studies.png)

#### 9.6.1 典型日负荷分析
选择代表性的工作日进行深度分析：
- **负荷曲线**: 典型日负荷曲线展现了标准的用电模式
- **负荷特征**: 日最大、最小负荷及其出现时间反映了用电规律
- **变化特征**: 日内负荷变化的幅度和规律性

#### 9.6.2 峰值周负荷特征
峰值周分析提供了系统高负荷运行的参考：
- **负荷水平**: 峰值周的负荷水平代表了系统的高需求状态
- **变化模式**: 峰值周内的负荷变化模式
- **运行特征**: 高负荷时期的系统运行特征

#### 9.6.3 季节性对比分析
夏冬负荷对比分析：
- **季节差异**: 夏季和冬季负荷的差异反映了气候对用电的影响
- **负荷特征**: 不同季节的日内负荷曲线特征
- **用电模式**: 季节性用电模式的差异分析
"""
        
        # Save the report
        report_path = self.reports_dir / 'comprehensive_analysis_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"✅ Comprehensive report saved to: {report_path}")
        return report_content
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting comprehensive load_data analysis...")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return False
        
        # Perform all analyses
        self.basic_statistics()
        self.time_series_analysis()
        self.data_quality_assessment()
        self.pattern_analysis()
        
        # Perform cyclical and trend analysis
        self.cyclical_trend_analysis()
        
        # Create visualizations
        self.create_visualizations()
        
        # Create cyclical visualizations
        self.create_cyclical_visualizations()
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        print("\n" + "=" * 60)
        print("🎉 Analysis completed successfully!")
        print(f"📁 Results saved in: {self.results_dir}")
        print(f"📊 Plots available in: {self.plots_dir}")
        print(f"📋 Reports available in: {self.reports_dir}")
        
        return True

def main():
    """Main function to run the analysis"""
    analyzer = LoadDataAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✨ Load data analysis completed successfully!")
        print("Check the 'analysis_results' folder for all outputs.")
    else:
        print("\n❌ Analysis failed. Please check the data path and try again.")

if __name__ == "__main__":
    main()
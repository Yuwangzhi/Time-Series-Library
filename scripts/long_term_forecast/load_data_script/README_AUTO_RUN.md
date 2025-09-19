# Load_Data 自动运行脚本使用说明

我已经为你创建了三个不同版本的自动运行脚本，可以一次性运行所有load_data模型的实验：

## 脚本列表

### 1. `run_all_load_data_models.sh` (推荐)
**Linux/Mac/Windows Git Bash 使用**
```bash
# 在 load_data_script 目录下运行
bash run_all_load_data_models.sh

# 或者从项目根目录运行
bash ./scripts/long_term_forecast/load_data_script/run_all_load_data_models.sh
```

### 2. `run_all_load_data_models.bat`
**Windows 命令行使用**
```cmd
# 在 load_data_script 目录下运行
run_all_load_data_models.bat

# 或者双击运行
```

### 3. `run_all_load_data_models.py` (最智能)
**Python 运行（跨平台）**
```bash
# 在 load_data_script 目录下运行
python run_all_load_data_models.py

# 或者从项目根目录运行
python ./scripts/long_term_forecast/load_data_script/run_all_load_data_models.py
```

## 功能特点

### 自动运行的模型 (14个)
1. Autoformer
2. Crossformer  
3. DLinear
4. FEDformer
5. Informer
6. iTransformer
7. MICN
8. Nonstationary_Transformer
9. PatchTST
10. SegRNN
11. TimeMixer
12. TimesNet
13. Transformer
14. TSMixer

### 每个模型的实验
- 96→96 预测
- 96→192 预测  
- 96→336 预测
- 96→720 预测

### 日志和结果
- **个别模型日志**: `./results/long_term_forecast_load_data_*/experiment_log.txt`
- **整体运行日志**: `load_data_all_models_log_YYYYMMDD_HHMMSS.txt`
- **模型检查点**: `./checkpoints/long_term_forecast_load_data_*/`

## 运行前准备

1. 确保在 Time-Series-Library 项目根目录
2. 确保 Python 环境已配置
3. 确保数据集已准备好在 `./dataset/load_data/hf_load_data/`

## 预计运行时间

- 每个模型约需要 30-60 分钟（取决于硬件）
- 总计 14 个模型约需要 8-14 小时
- 建议在有足够时间时运行（如过夜）

## 监控进度

### 实时监控
- 终端会显示当前运行的模型和进度
- 每个模型完成后会显示总结

### 日志监控
```bash
# 查看整体日志
tail -f load_data_all_models_log_*.txt

# 查看特定模型的实时日志
tail -f ./results/long_term_forecast_load_data_*/experiment_log.txt
```

## 中断和恢复

如果需要中断：
- 按 `Ctrl+C` 停止
- 可以手动运行剩余的单个模型脚本
- 或修改自动脚本，注释掉已完成的模型

## 结果分析

完成后，你将得到：
1. 14个模型 × 4个预测长度 = 56个实验结果
2. 每个实验的详细日志和指标
3. 所有模型的检查点文件
4. 预测结果文件

## 推荐使用顺序

1. **首次使用**: `run_all_load_data_models.py` (最稳定，有详细进度)
2. **Linux/Mac**: `run_all_load_data_models.sh`
3. **Windows**: `run_all_load_data_models.bat`

选择任意一个脚本运行即可！
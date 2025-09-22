@echo off
REM Auto-run all load_data model scripts (Windows batch version - 60-minute granularity)
REM This script will automatically execute all 60-minute granularity model scripts in sequence

echo ========================================================================
echo                AUTO-RUN ALL LOAD_DATA MODEL SCRIPTS (60-MIN)
echo ========================================================================
echo This script will run all available 60-minute granularity load_data model experiments in sequence.
echo Each model includes 4 prediction horizons: 96-^>96, 96-^>192, 96-^>336, 96-^>720
echo Data granularity: 60 minutes (1 hour)
echo Time interpretations:
echo - 96 time steps = 96 hours (96 × 60 min = 4 days)
echo - 192 time steps = 192 hours (192 × 60 min = 8 days)
echo - 336 time steps = 336 hours (336 × 60 min = 14 days)
echo - 720 time steps = 720 hours (720 × 60 min = 30 days)
echo Results and logs will be saved in respective ./results/ folders
echo ========================================================================
echo.

REM Record start time
echo Starting all 60-minute granularity model experiments at: %date% %time%
echo.

REM Create results directory if it doesn't exist
if not exist "..\..\..\results" mkdir "..\..\..\results"

REM Create overall log file (save to results directory)
set "OVERALL_LOG=..\..\..\results\load_data_60min_all_models_log_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt"
set "OVERALL_LOG=%OVERALL_LOG: =0%"
echo Overall experiment log: %OVERALL_LOG%
echo.

REM Initialize overall log
echo === LOAD_DATA 60-MINUTE ALL MODELS EXPERIMENT LOG === > "%OVERALL_LOG%"
echo Start Time: %date% %time% >> "%OVERALL_LOG%"
echo Data Granularity: 60 minutes (1 hour) >> "%OVERALL_LOG%"
echo. >> "%OVERALL_LOG%"

REM Define model scripts
set MODEL_NUM=0
set TOTAL_MODELS=14

REM Function to run each model
call :run_model "Autoformer_load_data_60min.sh" "Autoformer"
call :run_model "Crossformer_load_data_60min.sh" "Crossformer"
call :run_model "DLinear_load_data_60min.sh" "DLinear"
call :run_model "FEDformer_load_data_60min.sh" "FEDformer"
call :run_model "Informer_load_data_60min.sh" "Informer"
call :run_model "iTransformer_load_data_60min.sh" "iTransformer"
call :run_model "MICN_load_data_60min.sh" "MICN"
call :run_model "Nonstationary_Transformer_load_data_60min.sh" "Nonstationary_Transformer"
call :run_model "PatchTST_load_data_60min.sh" "PatchTST"
call :run_model "SegRNN_load_data_60min.sh" "SegRNN"
call :run_model "TimeMixer_load_data_60min.sh" "TimeMixer"
call :run_model "TimesNet_load_data_60min.sh" "TimesNet"
call :run_model "Transformer_load_data_60min.sh" "Transformer"
call :run_model "TSMixer_load_data_60min.sh" "TSMixer"

REM Summary
echo ========================================================================
echo 🎉 ALL 60-MINUTE GRANULARITY MODEL EXPERIMENTS COMPLETED!
echo ========================================================================
echo End Time: %date% %time%
echo Total Models Processed: %TOTAL_MODELS%
echo Data Granularity: 60 minutes (1 hour)
echo.
echo 📋 Results Summary:
echo - Individual model logs: ./results/long_term_forecast_load_data_60min_*/experiment_log.txt
echo - Overall experiment log: %OVERALL_LOG%
echo - Model checkpoints: ./checkpoints/long_term_forecast_load_data_60min_*/
echo.
echo Time interpretations for results:
echo - 96 time steps = 96 hours (96 × 60 min = 4 days)
echo - 192 time steps = 192 hours (192 × 60 min = 8 days)
echo - 336 time steps = 336 hours (336 × 60 min = 14 days)
echo - 720 time steps = 720 hours (720 × 60 min = 30 days)
echo.
echo You can now analyze the results from all %TOTAL_MODELS% 60-minute granularity models!
echo ========================================================================

REM Add summary to overall log
echo === EXPERIMENT COMPLETED === >> "%OVERALL_LOG%"
echo End Time: %date% %time% >> "%OVERALL_LOG%"
echo Total Models Processed: %TOTAL_MODELS% >> "%OVERALL_LOG%"
echo Data Granularity: 60 minutes (1 hour) >> "%OVERALL_LOG%"

pause
goto :eof

:run_model
set /a MODEL_NUM+=1
set "SCRIPT_NAME=%~1"
set "MODEL_NAME=%~2"

echo ========================================================================
echo 🔄 Running Model %MODEL_NUM%/%TOTAL_MODELS%: %MODEL_NAME% (60-min)
echo ========================================================================

echo [%MODEL_NUM%/%TOTAL_MODELS%] Starting %MODEL_NAME% (60-min) experiments... >> "%OVERALL_LOG%"
echo Model Start Time: %date% %time% >> "%OVERALL_LOG%"

REM Check if script exists and run it
if exist "%SCRIPT_NAME%" (
    echo Executing: bash %SCRIPT_NAME%
    bash "%SCRIPT_NAME%"
    
    if %errorlevel% equ 0 (
        echo ✅ %MODEL_NAME% (60-min) completed successfully at: %date% %time%
        echo [%MODEL_NUM%/%TOTAL_MODELS%] ✅ %MODEL_NAME% (60-min) completed successfully at: %date% %time% >> "%OVERALL_LOG%"
    ) else (
        echo ❌ %MODEL_NAME% (60-min) failed at: %date% %time%
        echo [%MODEL_NUM%/%TOTAL_MODELS%] ❌ %MODEL_NAME% (60-min) failed at: %date% %time% >> "%OVERALL_LOG%"
    )
) else (
    echo ❌ Script not found: %SCRIPT_NAME%
    echo [%MODEL_NUM%/%TOTAL_MODELS%] ❌ Script not found: %SCRIPT_NAME% >> "%OVERALL_LOG%"
)

echo. >> "%OVERALL_LOG%"
echo.
echo ------------------------------------------------------------------------
echo Completed %MODEL_NUM% out of %TOTAL_MODELS% models
set /a REMAINING=%TOTAL_MODELS%-%MODEL_NUM%
echo Remaining: %REMAINING% models
echo ------------------------------------------------------------------------
echo.
goto :eof
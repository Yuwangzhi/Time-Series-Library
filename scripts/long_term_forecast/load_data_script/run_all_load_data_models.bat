@echo off
REM Auto-run all load_data model scripts (Windows batch version)
REM This script will automatically execute all model scripts in sequence

echo ========================================================================
echo                     AUTO-RUN ALL LOAD_DATA MODEL SCRIPTS
echo ========================================================================
echo This script will run all available load_data model experiments in sequence.
echo Each model includes 4 prediction horizons: 96-^>96, 96-^>192, 96-^>336, 96-^>720
echo Results and logs will be saved in respective ./results/ folders
echo ========================================================================
echo.

REM Record start time
echo Starting all model experiments at: %date% %time%
echo.

REM Create results directory if it doesn't exist
if not exist "..\..\..\results" mkdir "..\..\..\results"

REM Create overall log file (save to results directory)
set "OVERALL_LOG=..\..\..\results\load_data_all_models_log_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.txt"
set "OVERALL_LOG=%OVERALL_LOG: =0%"
echo Overall experiment log: %OVERALL_LOG%
echo.

REM Initialize overall log
echo === LOAD_DATA ALL MODELS EXPERIMENT LOG === > "%OVERALL_LOG%"
echo Start Time: %date% %time% >> "%OVERALL_LOG%"
echo. >> "%OVERALL_LOG%"

REM Define model scripts
set MODEL_NUM=0
set TOTAL_MODELS=14

REM Function to run each model
call :run_model "Autoformer_load_data.sh" "Autoformer"
call :run_model "Crossformer_load_data.sh" "Crossformer"
call :run_model "DLinear_load_data.sh" "DLinear"
call :run_model "FEDformer_load_data.sh" "FEDformer"
call :run_model "Informer_load_data.sh" "Informer"
call :run_model "iTransformer_load_data.sh" "iTransformer"
call :run_model "MICN_load_data.sh" "MICN"
call :run_model "Nonstationary_Transformer_load_data.sh" "Nonstationary_Transformer"
call :run_model "PatchTST_load_data.sh" "PatchTST"
call :run_model "SegRNN_load_data.sh" "SegRNN"
call :run_model "TimeMixer_load_data.sh" "TimeMixer"
call :run_model "TimesNet_load_data.sh" "TimesNet"
call :run_model "Transformer_load_data.sh" "Transformer"
call :run_model "TSMixer_load_data.sh" "TSMixer"

REM Summary
echo ========================================================================
echo 🎉 ALL MODEL EXPERIMENTS COMPLETED!
echo ========================================================================
echo End Time: %date% %time%
echo Total Models Processed: %TOTAL_MODELS%
echo.
echo 📋 Results Summary:
echo - Individual model logs: ./results/long_term_forecast_load_data_*/experiment_log.txt
echo - Overall experiment log: %OVERALL_LOG%
echo - Model checkpoints: ./checkpoints/long_term_forecast_load_data_*/
echo.
echo You can now analyze the results from all %TOTAL_MODELS% models!
echo ========================================================================

REM Add summary to overall log
echo === EXPERIMENT COMPLETED === >> "%OVERALL_LOG%"
echo End Time: %date% %time% >> "%OVERALL_LOG%"
echo Total Models Processed: %TOTAL_MODELS% >> "%OVERALL_LOG%"

pause
goto :eof

:run_model
set /a MODEL_NUM+=1
set "SCRIPT_NAME=%~1"
set "MODEL_NAME=%~2"

echo ========================================================================
echo 🔄 Running Model %MODEL_NUM%/%TOTAL_MODELS%: %MODEL_NAME%
echo ========================================================================

echo [%MODEL_NUM%/%TOTAL_MODELS%] Starting %MODEL_NAME% experiments... >> "%OVERALL_LOG%"
echo Model Start Time: %date% %time% >> "%OVERALL_LOG%"

REM Check if script exists and run it
if exist "%SCRIPT_NAME%" (
    echo Executing: bash %SCRIPT_NAME%
    bash "%SCRIPT_NAME%"
    
    if %errorlevel% equ 0 (
        echo ✅ %MODEL_NAME% completed successfully at: %date% %time%
        echo [%MODEL_NUM%/%TOTAL_MODELS%] ✅ %MODEL_NAME% completed successfully at: %date% %time% >> "%OVERALL_LOG%"
    ) else (
        echo ❌ %MODEL_NAME% failed at: %date% %time%
        echo [%MODEL_NUM%/%TOTAL_MODELS%] ❌ %MODEL_NAME% failed at: %date% %time% >> "%OVERALL_LOG%"
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
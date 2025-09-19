#!/bin/bash

# Auto-run all load_data model scripts
# This script will automatically execute all model scripts in sequence
# Each model will run 4 experiments: 96->96, 96->192, 96->336, 96->720

echo "========================================================================"
echo "                    AUTO-RUN ALL LOAD_DATA MODEL SCRIPTS"
echo "========================================================================"
echo "This script will run all available load_data model experiments in sequence."
echo "Each model includes 4 prediction horizons: 96->96, 96->192, 96->336, 96->720"
echo "Results and logs will be saved in respective ./results/ folders"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define all model scripts (excluding Multiple_models_load_data.sh and PatchTST_load_data_one_time.sh to avoid duplicates)
MODEL_SCRIPTS=(
    "Autoformer_load_data.sh"
    "Crossformer_load_data.sh"  
    "DLinear_load_data.sh"
    "FEDformer_load_data.sh"
    "Informer_load_data.sh"
    "iTransformer_load_data.sh"
    "MICN_load_data.sh"
    "Nonstationary_Transformer_load_data.sh"
    "PatchTST_load_data.sh"
    "SegRNN_load_data.sh"
    "TimeMixer_load_data.sh"
    "TimesNet_load_data.sh"
    "Transformer_load_data.sh"
    "TSMixer_load_data.sh"
)

# Record start time
START_TIME=$(date)
echo "🚀 Starting all model experiments at: $START_TIME"
echo ""

# Counter for tracking progress
TOTAL_MODELS=${#MODEL_SCRIPTS[@]}
CURRENT_MODEL=0

# Log file for overall progress
OVERALL_LOG="load_data_all_models_log_$(date +%Y%m%d_%H%M%S).txt"
echo "Overall experiment log: $OVERALL_LOG"
echo ""

# Create overall log file
echo "=== LOAD_DATA ALL MODELS EXPERIMENT LOG ===" > "$OVERALL_LOG"
echo "Start Time: $START_TIME" >> "$OVERALL_LOG"
echo "Total Models: $TOTAL_MODELS" >> "$OVERALL_LOG"
echo "Models to run: ${MODEL_SCRIPTS[*]}" >> "$OVERALL_LOG"
echo "" >> "$OVERALL_LOG"

# Function to run a single model script
run_model() {
    local script_name=$1
    local model_num=$2
    local total=$3
    
    echo "========================================================================"
    echo "🔄 Running Model $model_num/$total: $script_name"
    echo "========================================================================"
    
    # Extract model name from script name
    MODEL_NAME=$(echo "$script_name" | sed 's/_load_data\.sh//')
    
    echo "[$model_num/$total] Starting $MODEL_NAME experiments..." >> "$OVERALL_LOG"
    
    # Record model start time
    MODEL_START_TIME=$(date)
    echo "Model Start Time: $MODEL_START_TIME"
    echo "Model Start Time: $MODEL_START_TIME" >> "$OVERALL_LOG"
    
    # Run the script
    if [ -f "$SCRIPT_DIR/$script_name" ]; then
        echo "Executing: bash $SCRIPT_DIR/$script_name"
        bash "$SCRIPT_DIR/$script_name"
        
        # Check if the script ran successfully
        if [ $? -eq 0 ]; then
            MODEL_END_TIME=$(date)
            echo "✅ $MODEL_NAME completed successfully at: $MODEL_END_TIME"
            echo "[$model_num/$total] ✅ $MODEL_NAME completed successfully at: $MODEL_END_TIME" >> "$OVERALL_LOG"
        else
            MODEL_END_TIME=$(date)
            echo "❌ $MODEL_NAME failed at: $MODEL_END_TIME"
            echo "[$model_num/$total] ❌ $MODEL_NAME failed at: $MODEL_END_TIME" >> "$OVERALL_LOG"
        fi
    else
        echo "❌ Script not found: $script_name"
        echo "[$model_num/$total] ❌ Script not found: $script_name" >> "$OVERALL_LOG"
    fi
    
    echo "" >> "$OVERALL_LOG"
    echo ""
}

# Run all model scripts
for script in "${MODEL_SCRIPTS[@]}"; do
    CURRENT_MODEL=$((CURRENT_MODEL + 1))
    run_model "$script" "$CURRENT_MODEL" "$TOTAL_MODELS"
    
    # Add a separator between models
    echo "------------------------------------------------------------------------"
    echo "Completed $CURRENT_MODEL out of $TOTAL_MODELS models"
    echo "Remaining: $((TOTAL_MODELS - CURRENT_MODEL)) models"
    echo "------------------------------------------------------------------------"
    echo ""
    
    # Optional: Add a small delay between models (uncomment if needed)
    # sleep 2
done

# Record end time and summary
END_TIME=$(date)
echo "========================================================================"
echo "🎉 ALL MODEL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Start Time: $START_TIME"
echo "End Time: $END_TIME"
echo "Total Models Processed: $TOTAL_MODELS"
echo ""
echo "📋 Results Summary:"
echo "- Individual model logs: ./results/long_term_forecast_load_data_*/experiment_log.txt"
echo "- Overall experiment log: $OVERALL_LOG"
echo "- Model checkpoints: ./checkpoints/long_term_forecast_load_data_*/"
echo ""
echo "You can now analyze the results from all $TOTAL_MODELS models!"
echo "========================================================================"

# Add summary to overall log
echo "=== EXPERIMENT COMPLETED ===" >> "$OVERALL_LOG"
echo "End Time: $END_TIME" >> "$OVERALL_LOG"
echo "Total Models Processed: $TOTAL_MODELS" >> "$OVERALL_LOG"
echo "" >> "$OVERALL_LOG"
echo "Individual model results available in ./results/ folders" >> "$OVERALL_LOG"
echo "Model checkpoints available in ./checkpoints/ folders" >> "$OVERALL_LOG"
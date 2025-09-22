#!/bin/bash

# Auto-run all load_data model scripts (60-minute granularity)
# This script will automatically execute all 60-minute granularity model scripts in sequence
# Each model will run 4 experiments: 96->96, 96->192, 96->336, 96->720

echo "========================================================================"
echo "                    AUTO-RUN ALL LOAD_DATA MODEL SCRIPTS (60-MIN)"
echo "========================================================================"
echo "This script will run all available 60-minute granularity load_data model experiments in sequence."
echo "Each model includes 4 prediction horizons: 96->96, 96->192, 96->336, 96->720"
echo "Data granularity: 60 minutes (1 hour)"
echo "Time interpretations:"
echo "- 96 time steps = 96 hours (96 × 60 min = 4 days)"
echo "- 192 time steps = 192 hours (192 × 60 min = 8 days)"
echo "- 336 time steps = 336 hours (336 × 60 min = 14 days)"
echo "- 720 time steps = 720 hours (720 × 60 min = 30 days)"
echo "Results and logs will be saved in respective ./results/ folders"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define all 60-minute model scripts
MODEL_SCRIPTS=(
    "Autoformer_load_data_60min.sh"
    "Crossformer_load_data_60min.sh"  
    "DLinear_load_data_60min.sh"
    "FEDformer_load_data_60min.sh"
    "Informer_load_data_60min.sh"
    "iTransformer_load_data_60min.sh"
    "MICN_load_data_60min.sh"
    "Nonstationary_Transformer_load_data_60min.sh"
    "PatchTST_load_data_60min.sh"
    "SegRNN_load_data_60min.sh"
    "TimeMixer_load_data_60min.sh"
    "TimesNet_load_data_60min.sh"
    "Transformer_load_data_60min.sh"
    "TSMixer_load_data_60min.sh"
)

# Record start time
START_TIME=$(date)
echo "🚀 Starting all 60-minute granularity model experiments at: $START_TIME"
echo ""

# Counter for tracking progress
TOTAL_MODELS=${#MODEL_SCRIPTS[@]}
CURRENT_MODEL=0

# Create results directory if it doesn't exist
RESULTS_DIR="$SCRIPT_DIR/../../../results"
mkdir -p "$RESULTS_DIR"

# Log file for overall progress (save to results directory)
OVERALL_LOG="$RESULTS_DIR/load_data_60min_all_models_log_$(date +%Y%m%d_%H%M%S).log"
echo "Overall experiment log: $OVERALL_LOG"
echo ""

# Create overall log file
echo "=== LOAD_DATA 60-MINUTE ALL MODELS EXPERIMENT LOG ===" > "$OVERALL_LOG"
echo "Start Time: $START_TIME" >> "$OVERALL_LOG"
echo "Total Models: $TOTAL_MODELS" >> "$OVERALL_LOG"
echo "Data Granularity: 60 minutes (1 hour)" >> "$OVERALL_LOG"
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
    MODEL_NAME=$(echo "$script_name" | sed 's/_load_data_60min\.sh//')
    
    echo "[$model_num/$total] Starting $MODEL_NAME (60-min) experiments..." >> "$OVERALL_LOG"
    
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
            echo "✅ $MODEL_NAME (60-min) completed successfully at: $MODEL_END_TIME"
            echo "[$model_num/$total] ✅ $MODEL_NAME (60-min) completed successfully at: $MODEL_END_TIME" >> "$OVERALL_LOG"
        else
            MODEL_END_TIME=$(date)
            echo "❌ $MODEL_NAME (60-min) failed at: $MODEL_END_TIME"
            echo "[$model_num/$total] ❌ $MODEL_NAME (60-min) failed at: $MODEL_END_TIME" >> "$OVERALL_LOG"
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
echo "🎉 ALL 60-MINUTE GRANULARITY MODEL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Start Time: $START_TIME"
echo "End Time: $END_TIME"
echo "Total Models Processed: $TOTAL_MODELS"
echo "Data Granularity: 60 minutes (1 hour)"
echo ""
echo "📋 Results Summary:"
echo "- Individual model logs: ./results/long_term_forecast_load_data_60min_*/experiment_log.txt"
echo "- Overall experiment log: $OVERALL_LOG"
echo "- Model checkpoints: ./checkpoints/long_term_forecast_load_data_60min_*/"
echo ""
echo "Time interpretations for results:"
echo "- 96 time steps = 96 hours (96 × 60 min = 4 days)"
echo "- 192 time steps = 192 hours (192 × 60 min = 8 days)"
echo "- 336 time steps = 336 hours (336 × 60 min = 14 days)"
echo "- 720 time steps = 720 hours (720 × 60 min = 30 days)"
echo ""
echo "You can now analyze the results from all $TOTAL_MODELS 60-minute granularity models!"
echo "========================================================================"

# Add summary to overall log
echo "=== EXPERIMENT COMPLETED ===" >> "$OVERALL_LOG"
echo "End Time: $END_TIME" >> "$OVERALL_LOG"
echo "Total Models Processed: $TOTAL_MODELS" >> "$OVERALL_LOG"
echo "Data Granularity: 60 minutes (1 hour)" >> "$OVERALL_LOG"
echo "" >> "$OVERALL_LOG"
echo "Individual model results available in ./results/ folders" >> "$OVERALL_LOG"
echo "Model checkpoints available in ./checkpoints/ folders" >> "$OVERALL_LOG"
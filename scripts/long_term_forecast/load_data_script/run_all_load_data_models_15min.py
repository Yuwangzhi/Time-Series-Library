#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-run all load_data model scripts (15-minute granularity)
This script will automatically execute all 15-minute granularity model scripts in sequence
Each model will run 4 experiments: 96->96, 96->192, 96->336, 96->720
"""

import os
import subprocess
import time
import datetime
from pathlib import Path

def main():
    print("=" * 72)
    print("            AUTO-RUN ALL LOAD_DATA MODEL SCRIPTS (15-MIN)")
    print("=" * 72)
    print("This script will run all available 15-minute granularity load_data model experiments in sequence.")
    print("Each model includes 4 prediction horizons: 96->96, 96->192, 96->336, 96->720")
    print("Data granularity: 15 minutes")
    print("Time interpretations:")
    print("- 96 time steps = 24 hours (96 × 15 min)")
    print("- 192 time steps = 48 hours (192 × 15 min)")
    print("- 336 time steps = 84 hours (336 × 15 min)")
    print("- 720 time steps = 180 hours (720 × 15 min)")
    print("Results and logs will be saved in respective ./results/ folders")
    print("=" * 72)
    print()

    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Define all 15-minute model scripts
    model_scripts = [
        "Autoformer_load_data_15min.sh",
        "Crossformer_load_data_15min.sh",
        "DLinear_load_data_15min.sh", 
        "FEDformer_load_data_15min.sh",
        "Informer_load_data_15min.sh",
        "iTransformer_load_data_15min.sh",
        "MICN_load_data_15min.sh",
        "Nonstationary_Transformer_load_data_15min.sh",
        "PatchTST_load_data_15min.sh",
        "SegRNN_load_data_15min.sh",
        "TimeMixer_load_data_15min.sh",
        "TimesNet_load_data_15min.sh",
        "Transformer_load_data_15min.sh",
        "TSMixer_load_data_15min.sh"
    ]
    
    # Record start time
    start_time = datetime.datetime.now()
    print(f"🚀 Starting all 15-minute granularity model experiments at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create results directory if it doesn't exist
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "../../../results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create overall log file (save to results directory)
    overall_log = os.path.join(results_dir, f"load_data_15min_all_models_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
    print(f"Overall experiment log: {overall_log}")
    print()
    
    # Initialize overall log
    with open(overall_log, 'w') as f:
        f.write("=== LOAD_DATA 15-MINUTE ALL MODELS EXPERIMENT LOG ===\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Granularity: 15 minutes\n")
        f.write(f"Total Models: {len(model_scripts)}\n")
        f.write(f"Models to run: {', '.join(model_scripts)}\n")
        f.write("\n")
    
    # Track results
    successful_models = []
    failed_models = []
    total_models = len(model_scripts)
    
    # Run each model script
    for i, script_name in enumerate(model_scripts, 1):
        print("=" * 72)
        print(f"🔄 Running Model {i}/{total_models}: {script_name}")
        print("=" * 72)
        
        # Extract model name
        model_name = script_name.replace('_load_data_15min.sh', '')
        
        # Record model start time
        model_start_time = datetime.datetime.now()
        print(f"Model Start Time: {model_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Log to overall log
        with open(overall_log, 'a') as f:
            f.write(f"[{i}/{total_models}] Starting {model_name} (15-min) experiments...\n")
            f.write(f"Model Start Time: {model_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Check if script exists
        script_path = script_dir / script_name
        if script_path.exists():
            print(f"Executing: bash {script_path}")
            
            try:
                # Run the script
                result = subprocess.run(
                    ['bash', str(script_path)],
                    cwd=script_dir.parent.parent.parent,  # Go back to Time-Series-Library root
                    capture_output=False,  # Allow output to be displayed in real-time
                    text=True
                )
                
                model_end_time = datetime.datetime.now()
                duration = model_end_time - model_start_time
                
                if result.returncode == 0:
                    print(f"✅ {model_name} (15-min) completed successfully at: {model_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Duration: {duration}")
                    successful_models.append(model_name)
                    
                    with open(overall_log, 'a') as f:
                        f.write(f"[{i}/{total_models}] ✅ {model_name} (15-min) completed successfully at: {model_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Duration: {duration}\n")
                else:
                    print(f"❌ {model_name} (15-min) failed at: {model_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Duration: {duration}")
                    failed_models.append(model_name)
                    
                    with open(overall_log, 'a') as f:
                        f.write(f"[{i}/{total_models}] ❌ {model_name} (15-min) failed at: {model_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Duration: {duration}\n")
                        
            except Exception as e:
                model_end_time = datetime.datetime.now()
                print(f"❌ {model_name} (15-min) error: {str(e)}")
                failed_models.append(model_name)
                
                with open(overall_log, 'a') as f:
                    f.write(f"[{i}/{total_models}] ❌ {model_name} (15-min) error: {str(e)}\n")
        else:
            print(f"❌ Script not found: {script_name}")
            failed_models.append(model_name)
            
            with open(overall_log, 'a') as f:
                f.write(f"[{i}/{total_models}] ❌ Script not found: {script_name}\n")
        
        # Add to log
        with open(overall_log, 'a') as f:
            f.write("\n")
        
        print()
        print("-" * 72)
        print(f"Completed {i} out of {total_models} models")
        print(f"Remaining: {total_models - i} models")
        print("-" * 72)
        print()
    
    # Final summary
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    
    print("=" * 72)
    print("🎉 ALL 15-MINUTE GRANULARITY MODEL EXPERIMENTS COMPLETED!")
    print("=" * 72)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Duration: {total_duration}")
    print(f"Total Models Processed: {total_models}")
    print(f"Data Granularity: 15 minutes")
    print(f"Successful: {len(successful_models)}")
    print(f"Failed: {len(failed_models)}")
    print()
    
    if successful_models:
        print("✅ Successful Models:")
        for model in successful_models:
            print(f"  - {model}")
        print()
    
    if failed_models:
        print("❌ Failed Models:")
        for model in failed_models:
            print(f"  - {model}")
        print()
    
    print("📋 Results Summary:")
    print("- Individual model logs: ./results/long_term_forecast_load_data_15min_*/experiment_log.txt")
    print(f"- Overall experiment log: {overall_log}")
    print("- Model checkpoints: ./checkpoints/long_term_forecast_load_data_15min_*/")
    print()
    print("Time interpretations for results:")
    print("- 96 time steps = 24 hours (96 × 15 min)")
    print("- 192 time steps = 48 hours (192 × 15 min)")
    print("- 336 time steps = 84 hours (336 × 15 min)")
    print("- 720 time steps = 180 hours (720 × 15 min)")
    print()
    print(f"You can now analyze the results from all {len(successful_models)} successful 15-minute granularity models!")
    print("=" * 72)
    
    # Add summary to overall log
    with open(overall_log, 'a') as f:
        f.write("=== EXPERIMENT COMPLETED ===\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration}\n")
        f.write(f"Total Models Processed: {total_models}\n")
        f.write(f"Data Granularity: 15 minutes\n")
        f.write(f"Successful: {len(successful_models)}\n")
        f.write(f"Failed: {len(failed_models)}\n")
        f.write("\n")
        
        if successful_models:
            f.write("Successful Models:\n")
            for model in successful_models:
                f.write(f"  - {model}\n")
            f.write("\n")
        
        if failed_models:
            f.write("Failed Models:\n")
            for model in failed_models:
                f.write(f"  - {model}\n")
            f.write("\n")
        
        f.write("Individual model results available in ./results/ folders\n")
        f.write("Model checkpoints available in ./checkpoints/ folders\n")

if __name__ == "__main__":
    main()
"""
Kaggle-compatible evolution runner
Upload this with your models to Kaggle Notebooks
GPU: P100 (16GB) - 30 hours/week FREE

Setup:
1. Go to https://www.kaggle.com/code
2. New Notebook → Upload this file
3. Settings → Accelerator → GPU P100
4. Add Dataset with your models (upload as .zip)
5. Run!
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np

# Check GPU
import subprocess
print("GPU Info:")
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)

# Install OpenVINO
print("\nInstalling OpenVINO...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "openvino"], check=True)

import openvino as ov

# Configuration - Adjust paths based on your Kaggle dataset
DATASET_PATH = "/kaggle/input/cyberspore-models"  # Your uploaded dataset name
DENSE_MODEL_PATH = f"{DATASET_PATH}/gemma_ir/openvino_model.xml"
TSSN_MODEL_PATH = f"{DATASET_PATH}/gemma_ir_tssn/openvino_model.xml"
EXT_PATH = f"{DATASET_PATH}/openvino_tssn_extension.so"  # If you have Linux build
OUTPUT_DIR = "/kaggle/working"  # Where Kaggle saves outputs

# Evolution parameters
MAX_ITERATIONS = 1000
MUTATION_RATE = 0.1  # 10% neurons per iteration
CHECKPOINT_INTERVAL = 50  # Save every 50 iterations
DEVICE = "GPU"

def get_inputs(seq_len=32):
    """Generate test inputs."""
    return {
        "input_ids": np.random.randint(0, 256000, (1, seq_len), dtype=np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "position_ids": np.arange(seq_len, dtype=np.int64).reshape(1, seq_len),
        "beam_idx": np.zeros((1,), dtype=np.int32)
    }

def log_progress(message, log_file="evolution_log.txt"):
    """Log to file and console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(os.path.join(OUTPUT_DIR, log_file), "a") as f:
        f.write(log_msg + "\n")

def main():
    log_progress("="*80)
    log_progress("KAGGLE CYBERSPORE EVOLUTION")
    log_progress("="*80)
    
    # Initialize
    core = ov.Core()
    
    # Load extension if available
    if os.path.exists(EXT_PATH):
        core.add_extension(EXT_PATH)
        log_progress(f"✓ Loaded TSSN extension from {EXT_PATH}")
    else:
        log_progress(f"⚠ No TSSN extension found at {EXT_PATH}")
        log_progress("  Evolution will only optimize function selection, not kernel behavior")
    
    # Load models
    log_progress("\nLoading models...")
    log_progress(f"Teacher: {DENSE_MODEL_PATH}")
    log_progress(f"Student: {TSSN_MODEL_PATH}")
    
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    student_model = core.read_model(TSSN_MODEL_PATH)
    
    # Find TSSN layers
    tssn_ops = [op for op in student_model.get_ops() if op.get_type_name() == "CompositeTSSN"]
    log_progress(f"Found {len(tssn_ops)} TSSN layers to evolve")
    
    # Compile
    log_progress(f"\nCompiling for {DEVICE}...")
    compiled_teacher = core.compile_model(teacher_model, DEVICE)
    compiled_student = core.compile_model(student_model, DEVICE)
    log_progress("✓ Compilation complete")
    
    # Baseline evaluation
    log_progress("\n--- Baseline Evaluation ---")
    inputs = get_inputs()
    teacher_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in teacher_model.inputs]}
    student_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in student_model.inputs]}
    
    teacher_output = compiled_teacher(teacher_inputs)[0]
    student_output = compiled_student(student_inputs)[0]
    
    baseline_mse = float(np.mean((teacher_output - student_output)**2))
    log_progress(f"Baseline MSE: {baseline_mse:.6e}")
    
    best_mse = baseline_mse
    best_model = student_model
    
    # Evolution history
    history = {
        "baseline_mse": baseline_mse,
        "iterations": [],
        "improvements": 0,
        "total_mutations": 0
    }
    
    # Evolution loop
    log_progress("\n--- Starting Evolution ---")
    start_time = time.time()
    
    for iteration in range(1, MAX_ITERATIONS + 1):
        iter_start = time.time()
        
        # Select random layer to mutate
        target_op = tssn_ops[np.random.randint(0, len(tssn_ops))]
        layer_name = target_op.get_friendly_name()
        
        # Get function_ids constant (input 6)
        func_const = target_op.input(6).get_source_output().get_node()
        current_ids = func_const.get_data().copy()
        num_neurons = len(current_ids)
        
        # Mutate
        num_mutations = max(1, int(num_neurons * MUTATION_RATE))
        mutation_indices = np.random.choice(num_neurons, num_mutations, replace=False)
        mutated_ids = current_ids.copy()
        mutated_ids[mutation_indices] = np.random.randint(0, 16, num_mutations)
        
        # Create candidate model
        from openvino.runtime import opset10 as ops
        new_const = ops.constant(mutated_ids, dtype=np.int32)
        target_op.input(6).replace_source_output(new_const.output(0))
        
        # Test candidate
        try:
            compiled_candidate = core.compile_model(student_model, DEVICE)
            candidate_output = compiled_candidate(student_inputs)[0]
            candidate_mse = float(np.mean((teacher_output - candidate_output)**2))
            
            # Accept if improved
            if candidate_mse < best_mse:
                improvement = ((best_mse - candidate_mse) / best_mse) * 100
                best_mse = candidate_mse
                best_model = student_model
                history["improvements"] += 1
                history["total_mutations"] += num_mutations
                
                log_progress(f"Iter {iteration}: ✓ IMPROVEMENT on {layer_name}")
                log_progress(f"  MSE: {candidate_mse:.6e} (↓{improvement:.2f}%)")
                log_progress(f"  Mutated {num_mutations}/{num_neurons} neurons")
            else:
                # Revert
                revert_const = ops.constant(current_ids, dtype=np.int32)
                target_op.input(6).replace_source_output(revert_const.output(0))
                
                if iteration % 100 == 0:
                    log_progress(f"Iter {iteration}: No improvement (MSE: {candidate_mse:.6e})")
        
        except Exception as e:
            # Revert on error
            revert_const = ops.constant(current_ids, dtype=np.int32)
            target_op.input(6).replace_source_output(revert_const.output(0))
            log_progress(f"Iter {iteration}: ERROR - {str(e)[:100]}")
        
        # Record iteration
        iter_time = time.time() - iter_start
        history["iterations"].append({
            "iteration": iteration,
            "mse": best_mse,
            "time_seconds": iter_time
        })
        
        # Checkpoint
        if iteration % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(OUTPUT_DIR, f"checkpoint_iter{iteration}.xml")
            ov.save_model(best_model, checkpoint_path)
            log_progress(f"\n✓ Checkpoint saved: {checkpoint_path}")
            
            # Save history
            with open(os.path.join(OUTPUT_DIR, "evolution_history.json"), "w") as f:
                json.dump(history, f, indent=2)
    
    # Final results
    elapsed = time.time() - start_time
    log_progress("\n" + "="*80)
    log_progress("EVOLUTION COMPLETE")
    log_progress("="*80)
    log_progress(f"Total time: {elapsed/3600:.2f} hours")
    log_progress(f"Iterations: {MAX_ITERATIONS}")
    log_progress(f"Improvements: {history['improvements']}")
    log_progress(f"Total mutations: {history['total_mutations']}")
    log_progress(f"Initial MSE: {baseline_mse:.6e}")
    log_progress(f"Final MSE: {best_mse:.6e}")
    log_progress(f"Improvement: {((baseline_mse - best_mse)/baseline_mse)*100:.2f}%")
    
    # Save final model
    final_path = os.path.join(OUTPUT_DIR, "evolved_final.xml")
    ov.save_model(best_model, final_path)
    log_progress(f"\n✓ Final model saved: {final_path}")
    
    # Save results
    results = {
        "baseline_mse": baseline_mse,
        "final_mse": best_mse,
        "improvement_percent": ((baseline_mse - best_mse)/baseline_mse)*100,
        "total_iterations": MAX_ITERATIONS,
        "improvements": history["improvements"],
        "total_mutations": history["total_mutations"],
        "elapsed_hours": elapsed/3600,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(os.path.join(OUTPUT_DIR, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    log_progress("\n✓ All results saved to /kaggle/working/")
    log_progress("Download: evolved_final.xml, evolution_history.json, final_results.json")

if __name__ == "__main__":
    main()

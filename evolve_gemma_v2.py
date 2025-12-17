"""
Improved evolution script with logging, checkpointing, and progress tracking.
Optimized for GPU by using Dynamic Parameters to avoid recompilation.
"""
import openvino as ov
import numpy as np
import time
import os
import json
from datetime import datetime

# Configuration
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_MODEL_PATH = "gemma_ir_tssn/openvino_model.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "GPU"

# Evolution parameters
MAX_ITERATIONS = 1000
MUTATION_RATE = 0.1  # 10% of neurons per mutation
CHECKPOINT_INTERVAL = 10  # Save every N iterations
LOG_FILE = "evolution_progress.log"
CHECKPOINT_DIR = "evolution_checkpoints"
RESULTS_FILE = "evolution_results_v2.json"

def log_message(message, also_print=True):
    """Write message to log file and optionally print."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}\n"
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_line)
    
    if also_print:
        print(f"[{timestamp}] {message}")

def save_checkpoint(iteration, best_mse, student_model, results_history):
    """Save checkpoint of current state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Save model
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.xml")
    ov.save_model(student_model, checkpoint_path)
    
    # Save metadata
    metadata = {
        "iteration": iteration,
        "best_mse": float(best_mse),
        "timestamp": datetime.now().isoformat(),
        "results_history": results_history
    }
    
    metadata_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_iter_{iteration}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    log_message(f"Checkpoint saved: {checkpoint_path}")

def save_results(results_history):
    """Save complete results history."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_history, f, indent=2)

def get_inputs(seq_len=32):
    """Generate random test inputs."""
    input_ids = np.random.randint(0, 256000, (1, seq_len), dtype=np.int64)
    attention_mask = np.ones((1, seq_len), dtype=np.int64)
    position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, seq_len)
    beam_idx = np.zeros((1,), dtype=np.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "beam_idx": beam_idx
    }

def compile_and_infer(core, model, inputs, device="GPU"):
    """Compile model, run inference, then free memory."""
    compiled = core.compile_model(model, device)
    result = compiled(inputs)[0]
    del compiled  # Free GPU/CPU memory
    return result

def main():
    log_message("="*80)
    log_message("EVOLUTION V2: With Logging and Checkpointing (Dynamic Graph)")
    log_message("="*80)
    
    # Initialize
    core = ov.Core()
    log_message(f"Loading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})
    
    # Load models
    log_message("Loading Teacher (Dense)...")
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    
    log_message("Loading Student (TSSN)...")
    student_model = core.read_model(TSSN_MODEL_PATH)
    
    # Find TSSN Nodes and Replace Constants with Parameters
    tssn_ops = []
    layer_params = {} # Map layer_name -> Parameter Node
    current_func_ids = {} # Map layer_name -> numpy array of current IDs
    
    log_message("Refactoring Graph for Dynamic Execution...")
    
    for op in student_model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            tssn_ops.append(op)
            layer_name = op.get_friendly_name()
            
            # Get Function IDs Constant (Input 6)
            func_ids_input = op.input_value(6)
            func_ids_node = func_ids_input.get_node()
            
            # Extract initial data
            initial_data = func_ids_node.data.copy()
            current_func_ids[layer_name] = initial_data
            
            # Create Parameter
            param_name = f"{layer_name}_func_ids"
            new_param = ov.runtime.op.Parameter(ov.Type.i32, ov.Shape(initial_data.shape))
            new_param.set_friendly_name(param_name)
            
            # Replace Constant with Parameter
            op.input(6).replace_source_output(new_param.output(0))
            
            # Add parameter to model inputs
            student_model.add_parameters([new_param])
            layer_params[layer_name] = new_param
            
            log_message(f"  Converted {layer_name} input 6 to Parameter: {param_name}")

    log_message(f"Found {len(tssn_ops)} TSSN layers")
    
    # Generate test inputs
    inputs = get_inputs()
    teacher_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in teacher_model.inputs]}
    
    # Prepare Student Inputs (Standard + Dynamic Params)
    student_base_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in student_model.inputs]}
    
    # Compile Models ONCE
    log_message("\nCompiling Models (ONCE)...")
    
    log_message("  Compiling teacher...")
    compiled_teacher = core.compile_model(teacher_model, DEVICE)
    teacher_output = compiled_teacher(teacher_inputs)[0]
    log_message("  Teacher inference complete")
    
    log_message("  Compiling student...")
    compiled_student = core.compile_model(student_model, DEVICE)
    log_message("  Student compilation complete")
    
    # Initial Inference
    # Construct full input map
    full_student_inputs = student_base_inputs.copy()
    for layer_name, data in current_func_ids.items():
        param = layer_params[layer_name]
        full_student_inputs[param] = data
        
    student_output = compiled_student(full_student_inputs)[0]
    
    baseline_mse = np.mean((teacher_output - student_output)**2)
    log_message(f"Baseline MSE: {baseline_mse:.6e}")
    
    best_mse = baseline_mse
    improvements = 0
    failures = 0
    
    # Results tracking
    results_history = {
        "baseline_mse": float(baseline_mse),
        "iterations": []
    }
    
    # Evolution Loop
    log_message("\n--- Starting Evolutionary Cycle ---")
    log_message(f"Max iterations: {MAX_ITERATIONS}")
    log_message(f"Mutation rate: {MUTATION_RATE*100}%")
    log_message(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    
    infer_request = compiled_student.create_infer_request()
    
    try:
        for iteration in range(1, MAX_ITERATIONS + 1):
            iter_start = time.time()
            # log_message(f"\n{'='*80}")
            # log_message(f"Iteration {iteration}/{MAX_ITERATIONS}")
            
            # Select random layer and neurons to mutate
            target_op = tssn_ops[np.random.randint(0, len(tssn_ops))]
            layer_name = target_op.get_friendly_name()
            # log_message(f"  Target layer: {layer_name}")
            
            # Get current data
            original_data = current_func_ids[layer_name].copy()
            mutated_data = original_data.copy()
            
            # Mutate neurons
            mask = np.random.rand(*mutated_data.shape) < MUTATION_RATE
            num_mutations = np.sum(mask)
            mutated_data[mask] = np.random.randint(0, 5, size=num_mutations)
            
            # log_message(f"  Mutating {num_mutations}/{mutated_data.size} neurons ({100*num_mutations/mutated_data.size:.1f}%)")
            
            # Update Input Map
            # We only need to update the specific parameter for this layer
            # But infer_request needs all inputs. 
            # Optimization: Keep a persistent input dictionary or set inputs on request
            
            # Update the specific parameter in our tracking dict
            current_func_ids[layer_name] = mutated_data
            
            # Update the infer request input
            target_param = layer_params[layer_name]
            infer_request.set_tensor(target_param, ov.Tensor(mutated_data))
            
            # Ensure other inputs are set (only need to do this once if they don't change, but let's be safe)
            if iteration == 1:
                for k, v in student_base_inputs.items():
                    infer_request.set_tensor(k, ov.Tensor(v))
                for lname, data in current_func_ids.items():
                    if lname != layer_name:
                        infer_request.set_tensor(layer_params[lname], ov.Tensor(data))

            # Evaluate mutation
            # log_message("  Evaluating mutation...")
            eval_start = time.time()
            
            infer_request.infer()
            student_output_new = infer_request.get_output_tensor(0).data
            
            new_mse = np.mean((teacher_output - student_output_new)**2)
            
            eval_time = time.time() - eval_start
            
            mse_delta = new_mse - best_mse
            # log_message(f"  New MSE: {new_mse:.6e} (Δ: {mse_delta:+.6e})")
            # log_message(f"  Evaluation time: {eval_time:.4f}s")
            
            # Record iteration
            iter_result = {
                "iteration": iteration,
                "layer": layer_name,
                "mutations": int(num_mutations),
                "mse": float(new_mse),
                "mse_delta": float(mse_delta),
                "best_mse": float(best_mse),
                "kept": False,
                "eval_time": eval_time
            }
            
            # Selection: Keep if improvement
            if new_mse < best_mse:
                improvements += 1
                improvement_pct = ((best_mse - new_mse) / best_mse) * 100
                log_message(f"[{iteration}] ✅ IMPROVEMENT! {improvement_pct:.4f}% better (MSE: {new_mse:.6e}) - Time: {eval_time:.4f}s")
                best_mse = new_mse
                iter_result["kept"] = True
                
                # Save evolved model (Need to bake constants back in for saving)
                # We do this periodically or at the end, not every step to save time
            else:
                failures += 1
                # log_message(f"  ❌ No improvement - REVERTING")
                
                # Revert mutation in our tracking dict
                current_func_ids[layer_name] = original_data
                # Revert in infer request
                infer_request.set_tensor(target_param, ov.Tensor(original_data))
            
            iter_result["total_time"] = time.time() - iter_start
            results_history["iterations"].append(iter_result)
            
            # Progress summary
            if iteration % 10 == 0:
                success_rate = (improvements / iteration) * 100
                log_message(f"[{iteration}] Progress: {improvements} improvements / {failures} failures ({success_rate:.1f}% success)")
            
            # Periodic checkpoint
            if iteration % CHECKPOINT_INTERVAL == 0:
                # To save, we need to reconstruct the model with Constants
                # Or just save the parameters and reload later.
                # For now, let's save the parameter values to JSON
                save_results(results_history)
                
                # Save parameter dump
                param_dump = {k: v.tolist() for k, v in current_func_ids.items()}
                with open(os.path.join(CHECKPOINT_DIR, f"params_iter_{iteration}.json"), "w") as f:
                    json.dump(param_dump, f)
            
    except KeyboardInterrupt:
        log_message("\n⚠️  Evolution interrupted by user")
    except Exception as e:
        log_message(f"\n❌ ERROR: {str(e)}")
        raise
    finally:
        # Final save
        log_message("\n" + "="*80)
        log_message("FINAL RESULTS")
        log_message("="*80)
        log_message(f"Iterations completed: {len(results_history['iterations'])}")
        log_message(f"Total improvements: {improvements}")
        log_message(f"Total failures: {failures}")
        log_message(f"Final best MSE: {best_mse:.6e}")
        log_message(f"Total improvement: {((baseline_mse - best_mse) / baseline_mse * 100):.4f}%")
        
        # Bake parameters back into constants for final model save
        log_message("Baking parameters back into constants...")
        for op in student_model.get_ops():
            if op.get_type_name() == "CompositeTSSN":
                layer_name = op.get_friendly_name()
                if layer_name in current_func_ids:
                    data = current_func_ids[layer_name]
                    new_const = ov.runtime.op.Constant(data)
                    # Find the input that is the parameter
                    # It was input 6
                    op.input(6).replace_source_output(new_const.output(0))
        
        # Remove parameters from model inputs
        # This is tricky in OpenVINO runtime API, easier to just save the modified graph
        # The replace_source_output disconnects the parameter, so it should be pruned on save?
        # Explicitly removing parameters from the function signature might be needed
        # But saving the model usually traverses from results, so disconnected params are ignored.
        
        final_model_path = "gemma_ir_tssn/evolved_model_final.xml"
        ov.save_model(student_model, final_model_path)
        log_message(f"Final model saved to: {final_model_path}")
        
        save_results(results_history)
        
        log_message(f"\nResults saved to: {RESULTS_FILE}")
        log_message(f"Log saved to: {LOG_FILE}")
        log_message("="*80)

if __name__ == "__main__":
    main()

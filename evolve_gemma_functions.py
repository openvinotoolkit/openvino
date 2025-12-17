import openvino as ov
import numpy as np
import time
import os
import copy

# Configuration
DENSE_MODEL_PATH = "gemma_ir/openvino_model.xml"
TSSN_MODEL_PATH = "gemma_ir_tssn/openvino_model.xml"
EXT_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
GPU_CONFIG = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
DEVICE = "GPU"

def get_inputs(seq_len=32):
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

def main():
    core = ov.Core()
    
    # Load Extension
    print(f"Loading extension: {EXT_PATH}")
    core.add_extension(EXT_PATH)
    core.set_property("GPU", {"CONFIG_FILE": GPU_CONFIG})
    
    # 1. Load Teacher (Dense)
    print("Loading Teacher (Dense)...")
    teacher_model = core.read_model(DENSE_MODEL_PATH)
    compiled_teacher = core.compile_model(teacher_model, DEVICE)
    
    # 2. Load Student (TSSN)
    print("Loading Student (TSSN)...")
    student_model = core.read_model(TSSN_MODEL_PATH)
    
    # Find TSSN Nodes
    tssn_ops = []
    for op in student_model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            tssn_ops.append(op)
    
    print(f"Found {len(tssn_ops)} TSSN layers.")
    
    # Initial Compile
    print("Compiling Student...")
    compiled_student = core.compile_model(student_model, DEVICE)
    
    # Baseline Evaluation
    inputs = get_inputs()
    
    # Filter inputs
    teacher_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in teacher_model.inputs]}
    student_inputs = {k: v for k, v in inputs.items() if k in [i.any_name for i in student_model.inputs]}
    
    print("Running Baseline Inference...")
    res_teacher = compiled_teacher(teacher_inputs)[0]
    res_student = compiled_student(student_inputs)[0]
    
    baseline_mse = np.mean((res_teacher - res_student)**2)
    print(f"Baseline MSE: {baseline_mse:.6f}")
    
    best_mse = baseline_mse
    
    # Evolution Loop
    print("\n--- Starting Evolutionary Cycle ---")
    for iteration in range(1, 1000): # Increased iterations for full evolution
        print(f"\nIteration {iteration}:")
        
        # 1. Mutate: Select a random layer and random neurons
        target_op = tssn_ops[np.random.randint(0, len(tssn_ops))]
        print(f"Mutating layer: {target_op.get_friendly_name()}")
        
        # Get Function IDs Constant (Input 6)
        func_ids_input = target_op.input_value(6)
        func_ids_node = func_ids_input.get_node()
        
        # Get data (copy)
        # Note: In OpenVINO Python, getting data from Constant might return a view or copy.
        # We create a new Constant.
        original_data = func_ids_node.data.copy()
        mutated_data = original_data.copy()
        
        # Mutate 10% of neurons
        mask = np.random.rand(*mutated_data.shape) < 0.1
        mutated_data[mask] = np.random.randint(0, 5, size=np.sum(mask))
        
        # Create new Constant
        new_const = ov.runtime.op.Constant(mutated_data)
        
        # Replace in graph
        target_op.input(6).replace_source_output(new_const.output(0))
        
        # Recompile
        print("Recompiling...")
        start_time = time.time()
        compiled_student = core.compile_model(student_model, DEVICE)
        print(f"Compile time: {time.time() - start_time:.2f}s")
        
        # Evaluate
        res_student_new = compiled_student(student_inputs)[0]
        new_mse = np.mean((res_teacher - res_student_new)**2)
        print(f"New MSE: {new_mse:.6f} (Delta: {new_mse - best_mse:.6f})")
        
        if new_mse < best_mse:
            print("Improvement! Keeping mutation.")
            best_mse = new_mse
            # Save checkpoint
            ov.save_model(student_model, "gemma_ir_tssn/evolved_checkpoint.xml")
        else:
            print("No improvement. Reverting.")
            # Revert graph change
            old_const = ov.runtime.op.Constant(original_data)
            target_op.input(6).replace_source_output(old_const.output(0))
            # No need to recompile immediately if we just continue to next mutation, 
            # but strictly we should restore the compiled model state or just recompile next time.
            # For simplicity, we just revert the graph.

if __name__ == "__main__":
    main()

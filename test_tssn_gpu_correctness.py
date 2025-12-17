import openvino as ov
import numpy as np
import os
import sys

# Add current directory to path to import the model generator
sys.path.append(".")
import create_evolutionary_benchmark_model

# Configuration
MODEL_NAME = "evolved_tssn_benchmark"
MODEL_PATH = f"{MODEL_NAME}.xml"
CONFIG_FILE = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
EXTENSION_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
DEVICE = "GPU"

def test_correctness():
    print(f"Testing correctness of {MODEL_PATH} on {DEVICE}...")

    input_dim = 1024
    output_dim = 1024
    sparsity = 0.1
    func_ids = [0] * output_dim

    # Regenerate model to ensure it uses Parameters
    print("Regenerating model with Parameters...")
    indices, weights, sensitivity, counts, starts, function_ids = create_evolutionary_benchmark_model.save_model(input_dim, output_dim, sparsity, func_ids, MODEL_NAME)

    # Get the data directly
    # indices, weights, sensitivity, counts, starts, function_ids = create_evolutionary_benchmark_model.create_sparse_data(input_dim, output_dim, sparsity)
    
    print(f"Indices shape: {indices.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Counts shape: {counts.shape}")
    print(f"Starts shape: {starts.shape}")
    print(f"Function IDs shape: {function_ids.shape}")
    
    core = ov.Core()
    core.add_extension(EXTENSION_PATH)
    core.set_property("GPU", {"CONFIG_FILE": CONFIG_FILE})
    # Force FP32 inference to avoid FP16 conversion
    core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})

    model = core.read_model(MODEL_PATH)
    
    # Create random input
    input_data = np.random.rand(1, input_dim).astype(np.float32)
    
    # Prepare inputs
    inputs = {
        "input": input_data,
        "indices": indices,
        "weights": weights,
        "sensitivity": sensitivity,
        "counts": counts,
        "starts": starts,
        "function_ids": function_ids
    }
    
    # Run GPU inference
    print("Compiling model...")
    compiled_model = core.compile_model(model, DEVICE)
    print("Running inference...")
    # Use index 0 for output
    gpu_output = compiled_model(inputs)[0]
    
    # Run CPU reference
    print("Running CPU reference...")
    cpu_output = np.zeros((1, output_dim), dtype=np.float32)
    
    # Iterative reference for mixed functions
    for i in range(output_dim):
        func = function_ids[i]
        start = starts[i]
        count = counts[i]
        
        if count == 0:
            cpu_output[0, i] = 0.0
            continue
            
        neuron_syn_indices = slice(start, start + count)
        in_idxs = indices[0, neuron_syn_indices]
        ws = weights[neuron_syn_indices]
        
        vals = input_data[0, in_idxs] * ws
        
        if func == 0: # SUM
            res = np.sum(vals)
        elif func == 1: # MIN
            res = np.min(vals)
        elif func == 2: # MAX
            res = np.max(vals)
        elif func == 3: # T_WAVE
            res = np.sin(np.sum(vals))
        elif func == 4: # TERNARY_IF
            s = np.sum(vals)
            if s > 0.5: res = 1.0
            elif s < -0.5: res = -1.0
            else: res = 0.0
        else:
            res = 0.0
            
        cpu_output[0, i] = res
    
    print("Comparison:")
    # gpu_output is [1, 1024], flatten it
    gpu_flat = gpu_output.flatten()
    cpu_flat = cpu_output.flatten()
    
    print(f"GPU output[0] (INPUT0_OFFSET): {gpu_flat[0]}")
    print(f"GPU output[1] (INPUT4_OFFSET): {gpu_flat[1]}")
    print(f"GPU output[2] (OUTPUT0_OFFSET): {gpu_flat[2]}")
    print(f"GPU output[3] (Global Size): {gpu_flat[3]}")
    print(f"GPU output[4] (x[0]): {gpu_flat[4]}")
    print(f"GPU output[5] (counts[0]): {gpu_flat[5]}")
    print(f"GPU output[6] (indices[0]): {gpu_flat[6]}")
    print(f"GPU output[7] (weights[0]): {gpu_flat[7]}")
    print(f"GPU output[8] (sensitivity[0]): {gpu_flat[8]}")
    print(f"GPU output[9] (counts[0] direct): {gpu_flat[9]}")
    print(f"GPU output[10] (starts[0]): {gpu_flat[10]}")
    
    print(f"CPU output[0]: {cpu_flat[0]}")

    print("Expected:")
    print(f"x[0]: {input_data[0,0]}")
    print(f"indices[0]: {indices.flatten()[0]}")
    print(f"indices[1]: {indices.flatten()[1]}")
    print(f"weights[0]: {weights[0]}")
    print(f"weights[1]: {weights[1]}")
    print(f"counts[0]: {counts[0]}")
    print(f"starts[0]: {starts[0]}")

    diff = np.abs(gpu_output - cpu_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")

    if max_diff < 1e-3:
        print("SUCCESS: GPU output matches CPU reference!")
    else:
        print("FAILURE: Significant difference detected.")
        # Print some mismatches
        mismatches = np.where(diff > 1e-3)
        print(f"Number of mismatches: {len(mismatches[0])}")
        if len(mismatches[0]) > 0:
            idx = mismatches[0][0] # flatten index
            # If 2D, use tuple
            if len(gpu_output.shape) == 2:
                r, c = mismatches[0][0], mismatches[1][0]
                print(f"Example mismatch at index [{r},{c}]: GPU={gpu_output[r, c]}, CPU={cpu_output[r, c]}")
                print(f"Difference: {np.abs(gpu_output[r, c] - cpu_output[r, c])}")
            else:
                print(f"Example mismatch at index {idx}: GPU={gpu_flat[idx]}, CPU={cpu_flat[idx]}")

if __name__ == "__main__":
    test_correctness()

import openvino as ov
import numpy as np
import time
import os
import sys

# Add current directory to path
sys.path.append(".")
import create_evolutionary_benchmark_model

print(f"OpenVINO Location: {ov.__file__}")
print(f"OpenVINO Version: {ov.__version__}")

# Configuration
MODEL_NAME = "benchmark_tssn"
MODEL_PATH = f"{MODEL_NAME}.xml"
CONFIG_FILE = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
EXTENSION_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
DEVICE = "CPU"

def benchmark():
    print(f"Benchmarking {DEVICE}...")
    
    input_dim = 4096
    output_dim = 4096 # Typical FFN size
    sparsity = 0.98 # Extreme sparsity as requested
    
    # 1. Create TSSN Model
    print("Generating TSSN Model...")
    indices, weights, sensitivity, counts, starts, function_ids = create_evolutionary_benchmark_model.save_model(
        input_dim, output_dim, sparsity, [], MODEL_NAME
    )
    
    core = ov.Core()
    core.add_extension(EXTENSION_PATH)
    core.set_property("GPU", {"CONFIG_FILE": CONFIG_FILE})
    core.set_property("GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    
    model_tssn = core.read_model(MODEL_PATH)
    compiled_tssn = core.compile_model(model_tssn, DEVICE)
    
    # Prepare Inputs
    input_data = np.random.rand(1, input_dim).astype(np.float32)
    inputs_tssn = {
        "input": input_data,
        "indices": indices,
        "weights": weights,
        "sensitivity": sensitivity,
        "counts": counts,
        "starts": starts,
        "function_ids": function_ids
    }
    
    # Warmup
    print("Warming up TSSN...")
    for _ in range(10):
        compiled_tssn(inputs_tssn)
        
    # Measure TSSN
    n_iter = 2000  # Increased for stability
    start_time = time.time()
    for _ in range(n_iter):
        compiled_tssn(inputs_tssn)
    end_time = time.time()
    tssn_fps = n_iter / (end_time - start_time)
    print(f"TSSN FPS: {tssn_fps:.2f}")
    
    # 2. Create Dense MatMul Model (Baseline)
    print("\nGenerating Dense Baseline Model...")
    # We can use a simple MatMul model
    # Create a dummy model in memory
    param = ov.opset10.parameter([1, input_dim], np.float32, name="input")
    const_weights = ov.opset10.constant(np.random.rand(input_dim, output_dim).astype(np.float32))
    matmul = ov.opset10.matmul(param, const_weights, transpose_a=False, transpose_b=False)
    model_dense = ov.Model([matmul], [param], "dense_baseline")
    
    compiled_dense = core.compile_model(model_dense, DEVICE)
    
    # Warmup
    print("Warming up Dense...")
    for _ in range(10):
        compiled_dense([input_data])
        
    # Measure Dense
    n_iter_dense = 500 # Dense is slower, so fewer iterations needed for stability
    start_time = time.time()
    for _ in range(n_iter_dense):
        compiled_dense([input_data])
    end_time = time.time()
    dense_fps = n_iter_dense / (end_time - start_time)
    print(f"Dense FPS: {dense_fps:.2f}")
    
    print(f"\nSpeedup: {tssn_fps / dense_fps:.2f}x")

if __name__ == "__main__":
    benchmark()

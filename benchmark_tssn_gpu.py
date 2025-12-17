import openvino as ov
import numpy as np
import time
import os

# Configuration
# MODEL_PATH = "gemma_ir_tssn/openvino_model.xml"
MODEL_PATH = "evolved_tssn_benchmark.xml"
CONFIG_FILE = os.path.abspath("src/custom_ops/composite_tssn_gpu.xml")
EXTENSION_PATH = os.path.abspath("src/custom_ops/build/Release/openvino_tssn_extension.dll")
DEVICE = "GPU"
NUM_ITERATIONS = 50

def benchmark():
    print(f"Benchmarking {MODEL_PATH} on {DEVICE} with config {CONFIG_FILE}...")

    core = ov.Core()
    
    # Load the extension library
    print(f"Loading extension from: {EXTENSION_PATH}")
    core.add_extension(EXTENSION_PATH)
    
    # Load the extension config
    print(f"Loading GPU config from: {CONFIG_FILE}")
    core.set_property("GPU", {"CONFIG_FILE": CONFIG_FILE})

    # Read the model
    print("Reading model...")
    model = core.read_model(MODEL_PATH)

    # Create input data
    # Simple benchmark inputs for evolved_tssn_benchmark.xml
    inputs = {
        "input": np.random.rand(1, 1024).astype(np.float32)
    }

    # Debug: Print CompositeTSSN output shapes
    print("Checking CompositeTSSN output shapes...")
    for op in model.get_ops():
        if op.get_type_name() == "CompositeTSSN":
            print(f"Layer {op.get_friendly_name()}: Output Partial Shape: {op.get_output_partial_shape(0)}")

    # Compile the model
    print("Compiling model...")
    compiled_model = core.compile_model(model, DEVICE)

    # Warm-up
    print("Warming up...")
    for _ in range(5):
        compiled_model(inputs)

    # Benchmark loop
    print(f"Running {NUM_ITERATIONS} iterations...")
    start_time = time.time()
    for i in range(NUM_ITERATIONS):
        compiled_model(inputs)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{NUM_ITERATIONS}")
            
    end_time = time.time()
    
    total_time = end_time - start_time
    fps = NUM_ITERATIONS / total_time
    
    print(f"\nTotal time: {total_time:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# simple_leak_test.py
# A minimal, dependency-free script to test OpenVINO Python API memory management.

import gc
import os
import psutil
import openvino as ov

MODEL_DIR = "distilbert-ov-int8"
DEVICE = "CPU"
LOOP_COUNT = 20

def get_memory_usage_gb():
    """Returns the current process's memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# --- Main Execution ---
print("--- Minimal OpenVINO Memory Leak Test for Issue #31383 ---")
print(f"Initial memory usage: {get_memory_usage_gb():.3f} GB")

# Ensure the model exists before starting the loop
if not os.path.isdir(MODEL_DIR):
    print(f"[ERROR] Model directory not found at: {os.path.abspath(MODEL_DIR)}")
    print(f"[ERROR] Please run the `run_leak_test.sh` script to create it automatically.")
    exit(1)

model_path = os.path.join(MODEL_DIR, "openvino_model.xml")

for i in range(LOOP_COUNT):
    print(f"\n---> Iteration [{i+1}/{LOOP_COUNT}] <---")
    print("Creating Core, reading and compiling model...")
    
    # Core object creation
    core = ov.Core()
    # Read model from file
    model = core.read_model(model_path)
    # Compile model for the target device
    compiled_model = core.compile_model(model, DEVICE)
    
    print(f"Memory after loading: {get_memory_usage_gb():.3f} GB")

    print("Unloading model and Core object...")
    del compiled_model
    del model
    del core
    gc.collect()

    print(f"Memory after unloading: {get_memory_usage_gb():.3f} GB")

# --- THE FIX ---
# After all operations are complete, call ov.shutdown() to release
# all internal static/global resources held by the OpenVINO runtime.
print("\n---> Performing final cleanup by calling openvino.shutdown() <---")
ov.shutdown()
print("Cleanup complete.")

print(f"\nFinal memory usage after {LOOP_COUNT} loops and cleanup: {get_memory_usage_gb():.3f} GB")
print("--- Test Complete ---")

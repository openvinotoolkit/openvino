import gc
import os
import psutil
from optimum.intel.openvino import OVModelForVisualCausalLM

MODEL_PATH = "./Qwen-OV-INT8"
DEVICE = "CPU"
LOOP_COUNT = 50 # A small number is fine for GDB

def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

print(f"Initial memory usage: {get_memory_usage_gb():.3f} GB")

for i in range(LOOP_COUNT):
    print(f"\n--- Iteration {i + 1}/{LOOP_COUNT} ---")
    
    print("Loading and compiling model...")
    model = OVModelForVisualCausalLM.from_pretrained(MODEL_PATH, device=DEVICE)
    print(f"Memory after loading: {get_memory_usage_gb():.3f} GB")
    
    print("Unloading model...")
    del model
    gc.collect()
    print(f"Memory after unloading: {get_memory_usage_gb():.3f} GB")

print(f"\nFinal memory usage after {LOOP_COUNT} loops: {get_memory_usage_gb():.3f} GB")

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import psutil
from optimum.intel.openvino import OVModelForCausalLM

# NOTE: This test requires a pre-converted model.
# Use 'convert_model.py' or a similar tool to create this directory.
MODEL_PATH = "Qwen-OV-INT8"
DEVICE = "CPU"
LOOP_COUNT = 20 # A smaller count is sufficient for Valgrind to detect the leak.

def get_memory_usage_gb():
    """Returns the current process's memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# --- Main Execution ---
print("--- Memory Leak Test for Issue #31383 ---")

# Check for and report the MALLOC_TRIM_THRESHOLD_ setting per maintainer feedback.
trim_threshold = os.environ.get('MALLOC_TRIM_THRESHOLD_')
if trim_threshold:
    print(f"[INFO] MALLOC_TRIM_THRESHOLD_ is set to: {trim_threshold}")
else:
    print("[INFO] MALLOC_TRIM_THRESHOLD_ is not set. Using default allocator behavior.")

print(f"Initial memory usage: {get_memory_usage_gb():.3f} GB")

for i in range(LOOP_COUNT):
    print(f"\n---> Iteration [{i+1}/{LOOP_COUNT}] <---")
    print("Loading and compiling model...")
    model = OVModelForCausalLM.from_pretrained(MODEL_PATH, device=DEVICE)
    print(f"Memory after loading: {get_memory_usage_gb():.3f} GB")

    print("Unloading model...")
    del model
    gc.collect() # Force Python's garbage collector.

    print(f"Memory after unloading: {get_memory_usage_gb():.3f} GB")

print(f"\nFinal memory usage after {LOOP_COUNT} loops: {get_memory_usage_gb():.3f} GB")
print("--- Test Complete ---")
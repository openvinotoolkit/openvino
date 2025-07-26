# Memory Leak Test for Issue #31383

This directory contains a Python script designed to reproduce the memory leak identified in GitHub issue #31383.

## Description

The script `reproduce_leak_31383.py` repeatedly loads and unloads a pre-converted OpenVINO model in a loop. It monitors the process's memory usage after each unload cycle.

On a faulty build, the memory usage will consistently increase with each iteration, demonstrating a memory leak of C++ resources. Once the underlying bug in the OpenVINO Core or its Python bindings is fixed, the memory usage reported by this script should remain stable.

## How to Run

1.  First, create a pre-converted model (e.g., `Qwen-OV-INT8`).
2.  Set up the Python virtual environment and `PYTHONPATH`/`LD_LIBRARY_PATH`.
3.  Run the script: `python reproduce_leak_31383.py`

# Memory Leak Test for Issue #31383

This directory contains a Python script designed to reproduce the memory leak identified in GitHub Issue [#31383](https://github.com/openvinotoolkit/openvino/issues/31383).

## Description

The script `reproduce_leak_31383.py` repeatedly loads and unloads a pre-converted OpenVINO model in a loop.

## Analysis and Findings

Initial testing showed a steady increase in the process's Resident Set Size (RSS), suggesting a memory leak. However, as noted by project maintainers, RSS is not a definitive proof of a leak due to the behavior of system memory allocators (e.g., glibc's `malloc`).

To provide definitive proof, this test was run under **Valgrind**.

The Valgrind report shows that upon program exit, there are **393,984 bytes in 3 blocks** that are "still reachable." This indicates that memory was allocated by the program but never freed, which is a memory leak.

Crucially, the call stacks for these leaked blocks trace back to the **Python interpreter's internal memory management** (`Py_InitializeFromConfig`, `PyDict_New`, etc.), not to the core OpenVINO C++ libraries. This suggests the leak occurs at the Python bindings layer, where C++ objects are created but are not being properly tracked and destroyed by Python's garbage collector during the load/unload cycle.

## How to Run

**Prerequisites:**

1.  Create a pre-converted model (e.g., `Qwen-OV-INT8`).
2.  Set up the Python virtual environment and necessary environment variables (`PYTHONPATH`, `LD_LIBRARY_PATH`).
3.  Install Valgrind: `sudo apt-get install valgrind`

### Definitive Leak Detection with Valgrind

This is the recommended method to confirm the leak. It runs the script under Valgrind's memory leak detector.

```bash
# From the openvino root directory:
valgrind --leak-check=full --show-leak-kinds=all --log-file="valgrind_report.txt" $(which python) samples/python/test_utils/memory_leak/reproduce_leak_31383.py
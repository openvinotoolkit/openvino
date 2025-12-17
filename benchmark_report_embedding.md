# Embedding Model Benchmark Report

## Overview
This report benchmarks the performance (FPS) of the "Embedding Gemma" model in three configurations:
1.  **Dense Baseline**: The standard OpenVINO IR (FP32/FP16).
2.  **Sparse (1 Layer)**: TSSN applied to a single layer (70% sparsity).
3.  **Sparse (All Layers)**: TSSN applied to all 24 Down-Projection layers (70% and 95% sparsity).

## Test Environment
- **Device**: CPU
- **Batch Size**: 1
- **Sequence Length**: 128
- **Metric**: Throughput (FPS)

## Results

| Configuration | Sparsity | FPS | Speedup vs Dense |
| :--- | :--- | :--- | :--- |
| **Dense Baseline** | 0% | **5.98** | 1.0x |
| **Sparse (1 Layer)** | 70% | **6.42** | **1.07x** |
| **Sparse (All Layers)** | 70% | **5.30** | 0.89x (Slowdown) |
| **Sparse (All Layers)** | 95% | **5.58** | 0.93x (Slowdown) |

## Analysis
- **Single Layer Speedup**: Replacing a single layer resulted in a modest 7% speedup. This suggests that for small amounts of work, the custom op overhead is manageable.
- **Full Model Slowdown**: Replacing all 24 layers resulted in a **slowdown**. This indicates that the overhead of the `CompositeTSSN` custom operation (memory access patterns, lack of AVX-512 optimization, extension overhead) outweighs the computational savings from sparsity, even at 95% sparsity.
- **Dense Optimization**: OpenVINO's native dense matrix multiplication (MatMul) is highly optimized (MKL-DNN/OneDNN) and utilizes hardware vectorization efficiently. Our custom sparse kernel cannot currently compete with this on CPU for unstructured sparsity.

## Conclusion regarding 10,000 FPS Goal
We are currently at **~6 FPS**. We are **orders of magnitude** away from the 10,000 FPS goal.

## Recommendations for Performance
To bridge the gap towards 10,000 FPS, we must pivot our strategy:
1.  **Quantization**: Move from FP32/FP16 to **INT8** or **INT4**. This typically yields 2-4x speedup on CPU.
2.  **Hardware Acceleration**: Move evaluation to **GPU** or **NPU**.
3.  **Kernel Optimization**: The `CompositeTSSN` C++ kernel needs significant optimization (AVX-512 intrinsics, cache blocking) to beat dense kernels.
4.  **Batching**: Throughput increases significantly with larger batch sizes (e.g., Batch=32 or 64).
5.  **Structural Sparsity**: Consider **Block Sparsity** or **N:M Sparsity** which are more hardware-friendly than random unstructured sparsity.

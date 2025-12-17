# Gemma TSSN Evolution Report

## 1. Performance Benchmark
We compared the performance of the Dense Gemma model against the TSSN-injected version.

| Metric | Dense Baseline | TSSN Injected | Speedup |
| :--- | :--- | :--- | :--- |
| **FPS** | 1.08 | 1.40 | **1.30x** |
| **Avg Latency** | 925.65 ms | 711.86 ms | **-23%** |

The TSSN injection provides a significant **30% speedup** on the GPU, validating the efficiency of the sparse operations even before functional optimization.

## 2. Evolutionary Cycle (Metabolic War)
We successfully initiated the evolutionary training loop to optimize the `function_ids` of the TSSN layers. The goal is to recover accuracy (minimize MSE against the teacher model) while maintaining the sparsity benefits.

### Initial Run Results (3 Iterations)
- **Baseline MSE**: 2.061191
- **Iteration 1**: 
  - Mutated Layer: `layers.8.mlp.down_proj`
  - **New MSE**: 2.057249 (Delta: -0.003942)
  - **Result**: Improvement! Mutation kept.
- **Iteration 2**:
  - Mutated Layer: `layers.2.mlp.down_proj`
  - Result: No improvement (Reverted).
- **Iteration 3**:
  - Mutated Layer: `layers.10.mlp.down_proj`
  - Result: No improvement (Reverted).

### Conclusion
The evolutionary mechanism is functional. The genetic algorithm is capable of finding mutations that reduce the error. The system is ready for a longer training run to fully recover the model's capabilities.

# Project Cyberspore: Phase 5 Carrying Capacity Report

## Objective
Determine the "Carrying Capacity" of the Cyberspore architecture by pushing sparsity beyond 50% until unrecoverable accuracy loss occurs (MTEB Gate Lock).

## Methodology
1.  **Models**: `EmbeddingGemma-300M` pruned at 60%, 70%, 80%, and 90% using **Magnitude Pruning**.
2.  **Metric**: Mean Squared Error (MSE) vs. Oracle (Original Model).
3.  **Healing**: 2000 Epochs of "Metabolic War" (Random Mutation).
4.  **Acceleration**: Attempted `SPARSE_WEIGHTS_DECOMPRESSION_RATE` (Result: Unsupported/Failed).

## Results

| Sparsity | Baseline MSE | Final MSE | Recovery % | Status |
| :--- | :--- | :--- | :--- | :--- |
| **60%** | 0.2678 | 0.2671 | **0.27%** | Stable |
| **70%** | 0.3273 | 0.3265 | **0.24%** | Strained |
| **80%** | 0.4605 | 0.4598 | **0.16%** | **CRITICAL FAILURE** |
| **90%** | 0.4759 | 0.4755 | **0.10%** | Terminal |

## Analysis

### 1. The MTEB Gate (Carrying Capacity Limit)
The system hits a "Wall" between **70% and 80% sparsity**.
-   **MSE Spike**: The error jumps by **~41%** (0.327 -> 0.460) when moving from 70% to 80%.
-   **Interpretation**: This indicates we have pierced the "Matryoshka Shells" and cut into the **Critical Semantic Core**. The outer 70% of weights were largely "noise" or fine-grained detail, but the remaining 30% are essential for the model's basic function.

### 2. Deep Infection Verification
-   **Diminishing Returns**: Recovery percentage drops linearly with sparsity (0.27% -> 0.10%).
-   **Lazy Virus Failure**: The TSSNs, initialized to zero and evolving via random mutation, fail to approximate the high-magnitude, non-linear functions of the inner core. The "Perturbation Recovery Time" effectively becomes infinite as the system fails to make meaningful progress.

### 3. Acceleration Status
-   **Sparse Weights Decompression**: The property `SPARSE_WEIGHTS_DECOMPRESSION_RATE` was rejected by the CPU plugin. This suggests either a version mismatch or that this optimization must be applied at the NNCF/Model Optimizer level, not runtime config.
-   **Bit-Packing**: Currently using `int8` (Soft Path). To achieve the target 8x-16x compression, we must implement the 2-bit packed format in the C++ kernel.

## Conclusion
The **Carrying Capacity** of the current Cyberspore architecture (Random Mutation TSSN) is approximately **70%**. Beyond this point, the host model suffers catastrophic semantic collapse that the current parasite cannot heal.

## Next Steps
1.  **Hard Acceleration**: Implement 2-bit packed storage in `tssn_op.cpp`.
2.  **Smart Infection**: Replace Random Mutation with a Gradient-Based or Heuristic-Guided evolution to tackle the "Inner Core".

# Workflow: Transformation

## Step 3: Create Test Cases from the New Model

Before modifying any transformation code, reproduce the failure as a unit test. These tests run without GPU hardware and give the fastest feedback.

### 3.1 Understand the New Routing Subgraph

Serialize the model (e.g. using `ov::pass::Serialize`) and trace the routing path from router logits to the `MOECompressed` input. Identify:

- **New operations** compared to existing patterns (e.g. `Sigmoid`, `Add(bias)`, `GatherElements`, `Slice`, `Add(eps)`)
- **New inputs** the fused op will need (e.g. `routing_bias`, `routing_eps`)
- **Shared tail** — the ScatterElementsUpdate → Transpose → Reshape → Unsqueeze chain is typically reused across routing variants

Existing patterns for reference:

**Softmax routing:**
```
MatMul → Softmax → TopK → ReduceSum → Divide(normalize)
                        ↘ ShapeOf → Gather → Unsqueeze ─┐
                                                         ├→ Concat → Broadcast
                                                         │
         ScatterElementsUpdate ← TopK_indices            │
              ↓                                          │
         Transpose → Reshape → Unsqueeze → MOECompressed
```

**Sigmoid+Bias routing:**
```
MatMul → Sigmoid → Add(bias) → TopK → Convert(i32) → GatherElements(sigmoid_out)
                                    ↘ ReduceSum → Add(eps) → Divide(normalize)
                                      → Slice → ScatterElementsUpdate
                                                     ↓
                                              Transpose → Reshape → Unsqueeze → MOECompressed
```

### 3.2 Add a Routing Type and Builder

**Files:**
- `src/tests/test_utils/common_test_utils/include/common_test_utils/node_builders/moe_builders.hpp`
- `src/tests/test_utils/common_test_utils/src/node_builders/moe_builders.cpp`

1. Add a new value to `MoERoutingType` (e.g. `SIGMOID_BIAS`)
2. Implement `build_<routing_name>_routing_subgraph()` returning the unsqueeze_moe output and topk_indices output
3. Dispatch on the new type in `initMoE3GeMMSubgraph()`

### 3.3 Write the Transformation Unit Test

**File:** `src/plugins/intel_gpu/tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp`

Build a test that:
1. Constructs the input graph (routing subgraph + `MOECompressed`) using the builder from 3.2
2. Runs `FuseMOE3GemmCompressed`
3. Compares against an expected graph containing `MOE3GemmFusedCompressed` with the correct routing type and input count

Parametrize over all routing types so existing cases stay covered:

```cpp
INSTANTIATE_TEST_SUITE_P(smoke, FuseMOE3GemmCompressedTest,
    ::testing::Values(MoERoutingType::SOFTMAX, MoERoutingType::SIGMOID_BIAS));
```

Run immediately — the new case is **expected to fail**, confirming you have reproduced the gap:

```bash
./scripts/build.sh ov_gpu_unit_tests
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*FuseMOE3GemmCompressed*'
```

## Step 4: Extend the Transformation and Make Tests Pass

Extend all layers touched by the new routing pattern until the unit test from Step 3.3 passes.

### 4.1 Extend the Internal Operation

**Files:**
- `src/plugins/intel_gpu/include/intel_gpu/op/moe_compressed.hpp` — add `RoutingType` enum value to `MOECompressed::Config`
- `src/plugins/intel_gpu/include/intel_gpu/op/moe_3gemm_fused_compressed.hpp` — document new inputs
- `src/plugins/intel_gpu/src/plugin/transformations/op/moe_compressed.cpp` — add `EnumNames` specialization and `visit_attributes` entry
- `src/plugins/intel_gpu/src/plugin/transformations/op/moe_3gemm_fused_compressed.cpp` — validate input count against routing type

### 4.2 Extend the Fusion Pattern

**Files:**
- `src/plugins/intel_gpu/src/plugin/transformations/fuse_moe_3gemm_compressed.cpp` / `.hpp`
- `src/plugins/intel_gpu/src/plugin/transformations/keep_moe_3gemm_const_precision.cpp`
- `src/plugins/intel_gpu/src/plugin/ops/moe.cpp`

Use `ov::pass::pattern::op::Or` to branch on the new routing subgraph. Detect the matched branch in the callback and push the extra inputs (e.g. `routing_bias`, `routing_eps`) into the fused op's argument list.

### 4.3 Extend Primitive Input Indices

**File:** `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_3gemm_base.hpp`

Add enum values for the new inputs (e.g. `ROUTING_BIAS = 11`, `ROUTING_EPS = 12`).

### 4.4 Verify and Loop

```bash
./scripts/build.sh ov_gpu_unit_tests
./scripts/run_tests.sh ov_gpu_unit_tests --gtest_filter='*FuseMOE3GemmCompressed*'
```

When the test passes, return to **Step 1** in [workflow-diagnosis.md](./workflow-diagnosis.md#step-1-run-with-matcher-logging) to confirm the transformation fires on the real model.

# FuseMOE3GemmCompressed Transformation

## Overview

`FuseMOE3GemmCompressed` is an OpenVINO matcher pass that fuses the MOE (Mixture of Experts) routing subgraph and compressed GEMM weights into a single `MOE3GemmFusedCompressed` primitive for execution on the Intel GPU plugin.

The pass handles two routing topologies and an optional per-expert or scalar routing scale.

## Matched pattern

```
hidden_state [→ Reshape]
    │
  MatMul(router weights)
    │
  ┌─┴─────────────────────────────────┐
  │ Softmax branch                    │ Sigmoid+bias branch
  │                                   │
  Softmax                           Sigmoid
    │                                 │  \
  TopK                              Add   \
  ├── values                      (bias)   \
  │     │                           │       \
  │   ReduceSum                   TopK       \
  │     │                           │         \
  │   Divide (norm)             Convert      GatherElements
  │     │                           │              │
  │  [Multiply]            (topk indices)      ReduceSum
  │     │                                          │
  └── [Convert]                               Add (eps)
        │                                         │
    (topk indices)                             Divide (norm)
                                                  │
                                             [Multiply]
                                                  │
                                         (topk indices)
  │
  Transpose → Unsqueeze
  │
MOECompressed(hidden_state, routing_weights, topk_indices,
              gate_wei, gate_scale, gate_zp,
              up_wei,   up_scale,   up_zp,
              down_wei, down_scale, down_zp
              [, shared_expert_weights...])
```

Nodes in `[brackets]` are optional.

## Routing variants

### Softmax (e.g. Mixtral, Gemma-4)

```
MatMul → Softmax → TopK → ReduceSum → Divide [→ Multiply] → Transpose → Unsqueeze
```

The optional `Multiply` absorbs a `routed_scaling_factor`. Two sub-variants exist:

| Sub-variant | Second input of Multiply | Handling |
|---|---|---|
| **Per-expert scale** (Gemma-4) | `Gather(Const[N], topk_idx)` | Folded statically into `w2_scale` (see below) |
| **Scalar scale** | Any scalar constant | Re-applied as a post-multiply on the fused op output |

### Sigmoid+bias (e.g. trinity-mini afmoe)

```
MatMul → Sigmoid → Add(bias) → TopK → Convert → GatherElements
                                                       │
                                                 ReduceSum → Add(eps) → Divide [→ Multiply]
                                                                                     │
                                                                              Transpose → Unsqueeze
```

The optional `Multiply` absorbs a scalar `routed_scaling_factor`, re-applied as a post-multiply on the fused op output.

## Per-expert scale folding (Gemma-4)

Gemma-4 has a `per_expert_scale` table (shape `[N]`, `f32`) looked up by routing indices:

```
per_expert_scale_const[N] → Gather(topk_indices) → Multiply(norm_weights, gathered_scales)
```

This scale **cannot** be moved after the `MOECompressed` output: after the expert accumulation `Σ wᵢ · yᵢ(x)`, the per-expert dimension is gone and the scale differs per expert.

Instead, the transformation folds it statically into `w2_scale` at graph-compile time:

```
w2_scale[N, hidden, groups, 1]  ×  per_expert_scale[N, 1, 1, 1]
    → folded_w2_scale[N, hidden, groups, 1]
```

Implementation uses a temporary `Unsqueeze + Multiply` subgraph evaluated by `ov::util::get_constant_from_source`, which leverages broadcasting without manual shape arithmetic. The result replaces `args[9]` (the `down_scale` input) before the fused node is constructed.

## Fused op inputs

`MOE3GemmFusedCompressed` receives inputs in this fixed order:

| Index | Content |
|---|---|
| 0 | hidden state (2D, flattened) |
| 1 | routing weights (MatMul output) |
| 2–4 | gate: weight, scale, zero-point |
| 5–7 | up: weight, scale, zero-point |
| 8–10 | down: weight, scale, zero-point |
| 11 | routing bias (SIGMOID_BIAS) or dummy `0` (SOFTMAX + shared expert) |
| 12 | routing eps (SIGMOID_BIAS) or dummy `0` (SOFTMAX + shared expert) |
| 13–22 | shared expert weights (when present) |

## Reshape handling

When `MOECompressed` receives the original 3D hidden state (not pre-flattened), the fused op works on the 2D flattened input taken from the routing subgraph. A `ShapeOf → Reshape` pair is inserted after the fused op to restore the original shape. This reshape is emitted **before** any post-multiply.

## Files

| File | Purpose |
|---|---|
| [fuse_moe_3gemm_compressed.hpp](../src/plugin/transformations/fuse_moe_3gemm_compressed.hpp) | Pass declaration |
| [fuse_moe_3gemm_compressed.cpp](../src/plugin/transformations/fuse_moe_3gemm_compressed.cpp) | Pass implementation |
| [fuse_moe_3gemm_compressed_test.cpp](../tests/unit/transformations/fuse_moe_3gemm_compressed_test.cpp) | Unit tests |

## See also

* [OpenVINO GPU Plugin](../README.md)
* [Graph optimization passes](./graph_optimization_passes.md)

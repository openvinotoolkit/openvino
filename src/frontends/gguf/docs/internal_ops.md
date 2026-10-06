# GGUF internal operations and serialization

GGUF normally emits public OpenVINO operations. It also uses shared core internal operations
for fused execution and a frontend placeholder for cache writes. These have different lifetimes.

## Conversion placeholder

[`ov::frontend::gguf::SetRows`](../include/openvino/frontend/gguf/set_rows_op.hpp) represents
`GGML_OP_SET_ROWS`. A caller-registered `GGUFMakeStateful` can consume cache writes before
the built-in `LowerSetRowsStateless` lowers remaining writes to ordinary graph operations.
`SetRows` must not survive normalization into the model returned by `convert()`.
See [pass registration order](extensions.md#register-normalization-passes) and
[`test_extensions.cpp`](../tests/test_extensions.cpp).

## Current internal ops emitted

These core `ov::op::internal` operations can remain in the converted model:

| Operation | GGML operation and selection | Alternative |
|---|---|---|
| `SelectiveSSM` | `SSM_SCAN`, Mamba 2 scalar decay | No frontend decomposition |
| `GatedDeltaNet` | `GATED_DELTA_NET`, scalar gate | `Loop` reference path for per-key gating or `force_ref`; incompatible with `split_outputs` |
| `GatherMatmul` | `MUL_MAT_ID`, eligible constant-backed expert weights | Generic Gather/MatMul path for other weights; selected by the converter, not a global portability switch |

Consult [`ssm_scan.cpp`](../src/op/ssm_scan.cpp),
[`gated_delta_net.cpp`](../src/op/gated_delta_net.cpp), and
[`mul_mat_id.cpp`](../src/op/mul_mat_id.cpp) for exact selection conditions.
Native GatedDeltaNet graphs request separate attention/state outputs with state layout
`[B,H_v,key_dim,value_dim]`; supplied ggml graphs retain their packed output/state convention.
The split form requires the fused scalar-gate path.

## Serialization and compiled-model caching

Core internal operations are outside the public IR opsets. Saving an IR can succeed with an
experimental operation version and still produce a file the ordinary IR frontend cannot reload.
Validate the save/read round trip; successful serialization alone is insufficient. If portable IR
is required, use supported public-op decompositions for every surviving internal operation.
An available per-op fallback is not a guarantee that the whole model has such a route.

Compiled-model export/import and caching use device-specific implementations. Do not infer their
support solely from IR serializability: check the target plugin, cache mode, and resulting graph,
then test export/import or cache reuse on that configuration. GGUF's in-process conversion paths
do not require an intermediate IR file.

## Adding an internal-op path

Prefer existing core operations with the required shape inference and target-plugin support.
Check evaluation/decomposition support explicitly rather than assuming every internal operation
has a portable fallback. Retain a public-op reference path where practical, test it against the
same oracle, and update the table with selection and serialization limits. Frontend placeholders
must have a normalization lowering; they are not device operations. Internal developer APIs may
change between releases.

# Workflow: Diagnosis

## Step 1: Run with Matcher Logging

Build `openvino_intel_gpu_plugin`, `openvino_ir_frontend`, and `benchmark_app`, then run the model with transformation logging enabled to observe whether the expected pipeline fires.

**Enable matcher logging** (see `src/common/transformations/docs/debug_capabilities/matcher_logging.md`):

```bash
# Build
./scripts/build.sh openvino_intel_gpu_plugin
./scripts/build.sh openvino_ir_frontend
./scripts/build.sh benchmark_app

# Run with logging enabled – redirect to .json for VS Code collapsible view
OV_MATCHER_LOGGING=true \
OV_MATCHERS_TO_LOG=FuseVectorizedMOE3GEMM,ConvertMOEToMOECompressed,FuseMOE3GemmCompressed,KeepMOE3GemmConstPrecision \
<build_dir>/bin/benchmark_app \
    -m <path_to_model.xml> \
    -d GPU \
    -niter 1 \
    -inference_precision f16 \
    2>matcher_log.json
```

For verbose output with per-node details:

```bash
OV_MATCHER_LOGGING=true OV_VERBOSE_LOGGING=true ...
```

## Step 2: Analyze the Logs

Check whether each pass in the pipeline fired successfully:

| Pass | Expected outcome |
|------|------------------|
| `FuseVectorizedMOE3GEMM` | `MOE` op appears in graph |
| `ConvertMOEToMOECompressed` | `MOECompressed` op appears |
| `FuseMOE3GemmCompressed` | `MOE3GemmFusedCompressed` op appears |
| `KeepMOE3GemmConstPrecision` | Weight constants stay u4 |

Look for `match succeeded` vs `match failed` entries for each matcher name. Confirm the runtime model contains `moe_3gemm_fused_compressed` layer type.

- If **all passes fired** → proceed to **Step 5** in [workflow-testing.md](./workflow-testing.md#step-5-extend-the-end-to-end-functional-test)
- If **any pass did not fire** → proceed to **Step 3** in [workflow-transformation.md](./workflow-transformation.md#step-3-create-test-cases-from-the-new-model)

## Known Issues

### `FuseVectorizedMOE3GEMM` does not fire — MatMul attribute mismatch

**Symptom:** Step 2 log shows `FuseVectorizedMOE3GEMM` never matching, even though the model visually contains the expected 3-GEMM MoE subgraph.

**Root cause:** The pattern in `FuseVectorizedMOE3GEMM` (`matmul_experts_fusion.cpp`) hard-codes attribute constraints on all three MatMuls:

```
gate_matmul:  transpose_a=false, transpose_b=true
up_matmul:    transpose_a=false, transpose_b=true
down_matmul:  transpose_a=false, transpose_b=true
```

If any MatMul has a different combination, the pattern match silently fails and the entire fusion chain never runs.

> ⛔ **This is not a skill bug — it is an IR conversion error.**
>
> **Required action:** Stop work on the GPU plugin and report back to the user that the model has been incorrectly converted to OpenVINO IR. All MoE expert MatMuls **must** be exported with `transpose_a=false` and `transpose_b=true`. The model conversion pipeline (frontend or export script) must be fixed before this skill can proceed. Do not attempt to work around this by modifying the pattern — the fix belongs at the IR level.

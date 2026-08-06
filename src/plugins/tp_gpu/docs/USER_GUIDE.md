# Tensor Parallel Plugin — User Guide

## What It Does

`TP_GPU` is an OpenVINO meta-device that runs a single
transformer model across two or more Intel discrete GPUs using
**Megatron-style tensor parallelism**:

- Q/K/V/gate/up MatMul weights are sharded column-wise across ranks —
  each rank computes a subset of attention heads / FFN channels.
- O/down MatMul weights are sharded row-wise; the partial sums are
  reduced across ranks via in-graph AllReduce.
- LayerNorm, RoPE, attention softmax, residual add, and embedding lookup
  are replicated on every rank.
- KV-cache shape constants are patched so each rank only stores its
  share of `num_kv_heads`.

The cross-device sum is performed directly on device USM via a shared
Level Zero context — no host round-trip and no copy through CPU memory.

## Requirements

- Two or more Intel discrete GPUs visible to the same Level Zero driver.
  Verified hardware: dual Intel Arc B580 (BMG) on PCIe 4.0 x8.
- Intel L0 driver (compute-runtime) with cross-device USM peer-copy
  support and the `zexCounterBasedEventCreate2` extension.
- OpenVINO built with `-DENABLE_TP_GPU=ON` (the default).
- A Llama-family OpenVINO IR. The plugin recognizes the standard
  `layers.{N}.self_attn.{q,k,v,o}_proj` and `layers.{N}.mlp.{gate,up,
  down}_proj` MatMul friendly names.

The plugin does **not** require MPI, NCCL, or any external collective
runtime. All synchronization is in-process across `std::async` rank
threads.

## Quick Start

### C++

```cpp
#include <openvino/openvino.hpp>

ov::Core core;
auto model = core.read_model("openvino_model.xml");

auto compiled = core.compile_model(model, "TP_GPU", {
    {"TP_SIZE", uint32_t(2)},
    // Optional: pin to specific devices.  Default: GPU.0, GPU.1, ...
    // {"DEVICE_IDS", std::vector<std::string>{"GPU.0", "GPU.1"}},
    {ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY},
    {ov::hint::inference_precision.name(), ov::element::f16},
});

auto request = compiled.create_infer_request();
request.set_input_tensor(input_tensor);
request.infer();
auto output = request.get_output_tensor();
```

### Python

```python
import openvino as ov

core = ov.Core()
model = core.read_model("openvino_model.xml")

compiled = core.compile_model(model, "TP_GPU", {
    "TP_SIZE": 2,
    "PERFORMANCE_HINT": "LATENCY",
    "INFERENCE_PRECISION_HINT": "f16",
})

request = compiled.create_infer_request()
request.infer({0: input_data})
output = request.get_output_tensor().data
```

## Configuration Properties

| Property                     | Type                          | Default                       | Description |
|------------------------------|-------------------------------|-------------------------------|-------------|
| `TP_SIZE`     | `uint32_t`                    | `2`                           | Number of ranks. Must equal `len(DEVICE_IDS)` when both are set. |
| `DEVICE_IDS`    | `std::vector<std::string>`    | `["GPU.0", "GPU.1", ...]`     | Explicit per-rank devices. |
| `COMMUNICATION_TIMEOUT_MS` | `uint32_t`       | `5000`                        | Longest a rank waits inside a collective before the group is aborted. `0` waits forever (debugging only). |

A collective needs all ranks to participate. If one rank never arrives or its
device work never completes, `COMMUNICATION_TIMEOUT_MS` is what turns that into
a thrown error instead of a frozen process: the group is marked failed, all
waiting ranks are released and each of them throws. The compiled model is not
reusable afterwards — the device queues may still hold unfinished work — so
every later inference on it fails immediately with the original reason.

Raise the value for very large payloads on slow interconnects; lower it if you
want a stuck rank reported sooner.

Any additional properties are passed through to the underlying
`intel_gpu` plugin used to compile each per-rank submodel. Common
choices: `PERFORMANCE_HINT`, `INFERENCE_PRECISION_HINT`,
`CACHE_DIR`, `NUM_STREAMS`.

### Forced Setting: Dynamic Quantization

The plugin overrides the `dynamic_quantization_group_size` hint to `0`
when the user does not set it. Reason: oneDNN's `i8 × i4` BRGEMM kernel
chooses different tilings for different K and N dimensions, and tensor
sharding changes those dimensions. The result is correctness drift
between the unsharded reference and the sharded execution at sequence
length ≥ 80 (`max_abs_diff` jumps from ~0.04 to >1.0 in measured runs).
Disabling dynamic quantization while keeping oneDNN itself active gives
f16 FC with the standard ~0.04 max_abs_diff. If you need quantization,
ensure your dataflow is robust to per-tile reorder before re-enabling.

## Best Practices

### Pre-warm to amortize plan rebuilds

The plugin caches per-collective execution plans. The first inference
at a given maximum sequence length pays a one-time cost (~16 ms across
all collectives for a 22-layer model) for plan construction; subsequent
inferences with shorter sequences re-record only (cheap), and identical
shapes reuse the plan entirely.

For LLM serving, run one dummy inference at the maximum expected
sequence length right after `compile_model`. Production traffic at any
lower length will then run on the cached, capacity-sized staging
buffers.

### Decode latency is the steady-state metric

For interactive serving, focus on **warm decode time per token** —
prefill and the first decode after a shape change pay one-time costs.
On TinyLlama-1.1B-int4 the validated dual-Arc system observes ~18 ms
per decode token in steady state, with about 4 ms attributable to the
44 AllReduce calls. Larger models amortize these collectives better.

### Tensor parallelism is not always faster

For very small models (≲ 1 B parameters in f16/i4), the per-call host
overhead and the cost of replicated norms/softmax dominate, and TP can
be **slower** than a single GPU. The break-even point on the validated
hardware is around 3 B parameters. Above 7 B the speedup is consistent
and grows with model size, capped by PCIe bandwidth on the AllReduce
critical path.

## Profiling

`request.get_profiling_info()` returns rank 0's per-op profile. Use it
to spot ops that dominate the per-step cost (typically attention SDPA
or the largest column/row-parallel MatMuls).

For collective-level profiling, set the env var `TP_PROF=N` (N = 2 ×
num_layers gives one report per inference). Output documents
record/exec/sync breakdown plus measured PCIe throughput per direction;
see [DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables-all-opt-in-via-env-var).

## Limitations

- **Architecture coverage.** The graph rewriter recognizes Llama-family
  layer naming. Other naming conventions (Mistral fused QKV, Falcon,
  GPT-J) need explicit pattern additions in
  `src/graph_rewriter.cpp::analyze`.
- **Divisibility.** `num_heads`, `num_kv_heads`, and `intermediate_size`
  must each be divisible by `TP_SIZE`. Non-divisible
  splits are not supported.
- **Single-process.** All ranks execute in the same process under
  `std::async`. Multi-process / multi-host configurations are out of
  scope; use NCCL/CCL-based stacks for those.
- **No model import/export, no user-supplied remote context.** The
  plugin always builds its own shared L0 context covering all ranks.
- **Dynamic quantization is force-disabled** (see above).
- **TP degree 2 is the validated production path.** Larger world sizes
  (3–4) build and run, but use a legacy funnel topology and have not
  been performance-tuned.

## Troubleshooting

### `TP_GPU` not in `core.get_available_devices()`

The plugin failed to load. Check:

- The shared object exists at `bin/intel64/Release/libopenvino_tp_gpu_plugin.so`.
- It is registered in `plugins.xml` (the build wires this up
  automatically — a stale `plugins.xml` from a previous build with
  `-DENABLE_TP_GPU=OFF` is a common cause; rebuild from
  scratch).
- `LD_LIBRARY_PATH` includes the runtime libs and the L0 loader is
  reachable.

### `[TP_GPU] Need at least 2 devices, got 1`

Only one Intel GPU is visible. Verify with `clinfo -l` or the L0
sysman tools (`zes_*`). On hosts where one GPU is reserved for display,
use `ZE_AFFINITY_MASK` only with care — it can hide the device from L0
entirely.

### Compilation fails on rank > 0 with an out-of-memory error

The current sharding halves weight footprints for column- and
row-parallel MatMuls but does **not** shard the embedding table, the
LM head, or activation memory. Effective memory per rank is roughly
`(weight_size / 2) + (full activation budget)`. If a single GPU does
not have enough VRAM for that, TP cannot fit it either.

### Outputs differ slightly from the single-GPU baseline

Expected, within typical FP rounding bounds. The baseline does the
attention/MLP MatMuls in a single accumulator order; TP splits them
across ranks and re-sums via the AllReduce kernel, which changes the
order of additions. Validated bound on the reference test harness:
`max_abs_diff` ≤ 0.04 with `INFERENCE_PRECISION_HINT=f16` and dynamic
quantization disabled. Top-K logit ordering matches.

### A second inference hangs

Always a resource-lifetime issue: device memory or a command list is
being freed/reset while still in flight. Capture the call with
`TP_DBG=1`; the last `[TP][L0]` line before the hang identifies the
operation. File the trace and the model dimensions when reporting.

## Pointers

- Internal architecture and the device-side AllReduce pipeline:
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Profiling, debugging recipes, environment variables, and how to add
  sharding patterns: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

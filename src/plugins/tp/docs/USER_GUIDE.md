# Tensor Parallel Plugin — User Guide

## What It Does

`TP` is an OpenVINO meta-device that runs a single transformer model across
two or more Intel discrete GPUs using **Megatron-style tensor parallelism**:

- Q/K/V/gate/up MatMul weights are sharded column-wise across ranks —
  each rank computes a subset of attention heads / FFN channels.
- O/down MatMul weights are sharded row-wise; the partial sums are
  reduced across ranks via in-graph AllReduce.
- The vocabulary projection is split along the vocabulary axis and the
  slices are gathered into rank 0, whose outputs the caller reads.
- LayerNorm, RoPE, attention softmax, residual add, and embedding lookup
  are replicated on every rank.
- KV-cache shape constants are patched so each rank only stores its
  share of `num_kv_heads`.

The cross-device sum is performed directly on device USM via a shared
Level Zero context — no host round-trip and no copy through CPU memory.

## Requirements

**Hardware and driver**

- Two or more Intel discrete GPUs, all served by the **same** Level Zero
  driver instance. The plugin builds one context spanning every rank device
  and refuses to start if the ranks land on different drivers.
- Cross-device USM peer copies and `zeContextCreateEx`.
- `zeCommandListImmediateAppendCommandListsExp` is optional: with it the
  collectives ride the model's own queue, without it they fall back to
  private queues automatically.

**Build**

- OpenVINO configured with `-DENABLE_TP=ON -DGPU_RT_TYPE=L0` (or `COMBINED`).
  The plugin is **off** by default, and the intel_gpu plugin has to include
  the Level Zero runtime rather than only the default OpenCL one -- the
  collectives run on Level Zero inside intel_gpu. CMake enforces this and
  fails configuration if the two disagree.

**Model**

- A decoder whose attention is `PagedAttentionExtension` or
  `ScaledDotProductAttention`, with the Q/K/V/O and MLP projections being
  MatMuls over constant-derived weights.
- The number of rank devices may not exceed the model's KV head count:
  attention is split by KV head, so there is nothing left to give the extra ranks.

## Is It Worth It

For small models the per-call dispatch and the cost of the replicated
norms and softmax dominate, and TP can be **slower** than a single GPU.
The larger the model, the better the collectives amortize; the ceiling is the
interconnect on the AllReduce critical path. Measure on your own model and
hardware rather than assuming a break-even point.

## Quick Start

### C++

```cpp
#include <openvino/openvino.hpp>

ov::Core core;
auto model = core.read_model("openvino_model.xml");

// One rank per listed device, in rank order.
auto compiled = core.compile_model(model, "TP:GPU.0,GPU.1");

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

compiled = core.compile_model(model, "TP:GPU.0,GPU.1")

request = compiled.create_infer_request()
request.infer({0: input_data})
output = request.get_output_tensor().data
```

## Configuration Properties

| Property                   | Type                       | Default     | Description                                                                                                                    |
|----------------------------|----------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------|
The rank devices are part of the device name, as for `HETERO`:
`"TP:GPU.0,GPU.1,GPU.2,GPU.3"` runs one rank per listed device, in rank order
(the list is `ov::device::priorities`). At least two devices of one device type
are required; `compile_model` throws otherwise. Whether the devices can run
the collectives is the device plugin's answer, asked before anything is
compiled (GPU: Level Zero devices of one driver with peer access between every
pair, each listed once). Only GPU devices can host ranks today.

| Property                   | Type       | Default | Description                                                                              |
|----------------------------|------------|---------|------------------------------------------------------------------------------------------|
| `TP_STRATEGY`              | string     | `COLUMN_ROW` | Partitioning strategy (Megatron-style column/row parallelism; the only value). |
| `COLLECTIVE_BACKEND`       | string     | `AUTO`  | Collective implementation of the rank devices' plugin, passed to it unchanged. GPU: `L0_P2P`; an unknown name is rejected with the valid ones. |
| `COMMUNICATION_TIMEOUT_MS` | `uint32_t` | `5000`  | Longest a rank waits inside a collective before the group is aborted. `0` waits forever. |
| `ov_internal_tp_comm_buffer_size` | `size_t` | `0` | Bytes of collective staging per rank, allocated once when the model is compiled. Larger messages are split into chunks. `0` lets the staging grow with the largest message. |

A collective needs all ranks to participate. If one rank never arrives or its
device work never completes, `COMMUNICATION_TIMEOUT_MS` is what turns that into
a thrown error instead of a frozen process: the group is marked failed, all
waiting ranks are released and each of them throws. The compiled model is not
reusable afterwards — the device queues may still hold unfinished work — so
every later inference on it fails immediately with the original reason.

Raise the value for very large payloads on slow interconnects; lower it if you
want a stuck rank reported sooner.

The collectives pass every message through a staging buffer on each rank.
Without `ov_internal_tp_comm_buffer_size` it grows to the largest message seen:
a prompt's prefill, which is the hidden size times the prompt length (128 MiB
per rank for an 8K-token prompt on Llama-3.1-8B), allocated when that prompt
first arrives. Setting it reserves the memory up front, before the KV cache is
sized, and caps it; a message that does not fit is sent in chunks. Splitting
into two chunks costs little; on four ranks, four chunks of an 8K-token prompt
cost about 6 % of the time to first token and eight about 11 %. Decode
messages are small and never split. On more than two ranks a chunked prompt
is summed in a different order, so the outputs may differ from the unchunked
run in the last bits, the same way on every run.

Any additional properties are passed through to the device plugin that
compiles each rank model. Properties for one
device or device type go through `ov::device::properties`, e.g.
`ov::device::properties("GPU.1", ...)` reaches rank 1 only. Every rank runs
one stream: `ov::num_streams` other than 1, here or in `ov::device::properties`,
is rejected.

The compiled model's context (`compiled.get_context()`) gives out host tensors
(`create_host_tensor()`, readable by every rank device); `create_tensor()` is
rejected, because a TP model has no single device. The compiled model reports
the rank count as `ov_internal_tp_world_size`.

## Profiling

`request.get_profiling_info()` returns **rank 0's** per-op profile, not an
average over ranks. Use it to spot ops that dominate the per-step cost,
typically attention SDPA or the largest column/row-parallel MatMuls. The
collectives themselves do not appear there as a separate entry; for those see
[DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables), which needs a
build with `-DENABLE_TP_DEBUG_CAPS=ON`.

## Model Caching

Compiling a large model once per rank is expensive, so `TP` supports
both explicit export/import and the standard `ov::cache_dir` flow:

```cpp
core.set_property(ov::cache_dir("/path/to/cache"));
// First call compiles every rank and writes one cache entry.
auto compiled = core.compile_model(model, "TP:GPU.0,GPU.1");
// Any later call with the same model and the same TP topology imports it.
```

Explicit form:

```cpp
std::stringstream blob;
compiled.export_model(blob);
auto imported = core.import_model(blob, "TP:GPU.0,GPU.1");
```

Things to know:

- **The blob is bound to the topology it was produced on.** It records the
  device of every rank, and rank `r`'s graph only makes sense on that device.
  Importing it against a different device set is rejected rather than silently
  retargeted. The device list is part of the cache hash, so different
  topologies get separate cache entries. The ranks write no cache entries of
  their own: the TP entry already contains them.
- **It is not portable.** Like any GPU blob, it is tied to the driver and
  hardware it was compiled on.
- **Weights are stored in full.** Weightless caching is not wired up, so the
  entry is roughly the size of the model.
- `compiled.get_property(ov::loaded_from_cache)` tells you which path was
  taken.

## Limitations

- **Architecture coverage.** The rewriter anchors on the attention op
  (`PagedAttentionExtension` or `ScaledDotProductAttention`) and walks back to
  the projections. What it does not handle is a different *shape*:
  fused QKV, or a weight fed from a live parameter rather than a constant.
- **Quantized weights.** A quantized weight can only be cut on a
  quantization-group boundary. A group size that does not line up with the
  split is rejected rather than silently mis-sliced.
- **Single-process.** All ranks execute in the same process on plugin-owned
  threads. Multi-process / multi-host configurations are out of scope.
- **One TP request at a time per device.** TP requests of every TP model in
  the process that shares a device with another run one after another.
- **No linear-attention models.** Models with linear-attention state
  (hybrid models) are rejected at compile time.
- **No user-supplied remote context.** The plugin always creates the rank
  contexts of its device group itself, so importing into a caller-provided
  context is rejected.
- **A failed collective is terminal.** After a timeout or an abort the
  compiled model refuses every later inference; it has to be rebuilt.
- **World sizes 2, 3 and 4** are covered by the functional suite. Nothing
  prevents larger ones, but they are untested.

## Troubleshooting

### `TP` not in `core.get_available_devices()`

The plugin failed to load, or was never built. Check:

- OpenVINO was configured with `-DENABLE_TP=ON -DGPU_RT_TYPE=L0` (or `COMBINED`).
  The plugin is off by default, so a stock build has no TP at all.
- The shared object exists at `bin/intel64/Release/libopenvino_tp_plugin.so`.
- It is registered in `plugins.xml` (the build wires this up
  automatically — a stale `plugins.xml` from a build without the plugin is a
  common cause; rebuild from scratch).
- `LD_LIBRARY_PATH` includes the runtime libs and the L0 loader is
  reachable.

### `[TP] Need at least 2 devices, got 1`

Only one Intel GPU is visible. Verify with `clinfo -l` or the L0 sysman tools.

### Compilation fails on rank > 0 with an out-of-memory error

The current sharding divides the weights of the column- and row-parallel
MatMuls and the vocabulary projection across ranks, but does **not** shard
the embedding table or activation memory. Effective memory per rank is
roughly `(sharded weights / N) + (unsharded weights) + (full activation
budget)`. If a single GPU does not have enough VRAM for that, TP cannot fit
it either.

### Outputs differ slightly from the single-GPU baseline

Expected, within floating-point rounding. The baseline does the
attention/MLP MatMuls in a single accumulator order; TP splits them
across ranks and re-sums via the AllReduce kernel, which changes the
order of additions. The functional suite pins the inference precision on
both sides and compares at `1e-2` for f16 and `1e-5` for f32.

### The first inference is much slower than the rest

Expected. Each collective records its command lists on first use and, unless
`ov_internal_tp_comm_buffer_size` fixes it, the staging arena grows to fit the
payload; both are one-time costs per shape.
A shorter sequence afterwards reuses what is already there, a longer one pays
again.

### `[TP] Warning: N KV heads do not divide evenly across M ranks`

Not an error: dimensions do not have to divide evenly by the number of ranks, and the
remainder goes to the first ranks, one unit each. The model runs.

It costs latency, though. Every layer ends in a collective, so the ranks that
got an extra head do more work while the rest wait at the barrier, and the
whole model moves at their pace. The percentage in the message is how much
longer those ranks have to work; the warning only appears past roughly 10%.

Choose a number of devices that divides the KV head count to make it go away.

### A collective times out or the group aborts

The error names the rank and the stage it gave up at. A rank that never
arrives usually means the surrounding application is doing something
uneven across ranks; device work that never completes usually means a
driver-level problem. Raise `COMMUNICATION_TIMEOUT_MS` for very large
payloads on slow interconnects. The compiled model is not reusable after an
abort.

For a trace of what each rank was doing, build with
`-DENABLE_TP_DEBUG_CAPS=ON` and set `OV_TP_VERBOSE=LOG_TRACE`.

## Pointers

- Internal architecture and the device-side AllReduce pipeline:
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Profiling, debugging recipes, environment variables, and how to add
  sharding patterns: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

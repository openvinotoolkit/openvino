# Tensor Parallel Plugin — User Guide

## What It Does

`TP` is an OpenVINO meta-device that runs a single transformer model across
two or more Intel discrete GPUs using **Megatron-style tensor parallelism**:

- Q/K/V/gate/up MatMul weights are sharded column-wise across ranks —
  each rank computes a subset of attention heads / FFN channels.
- O/down MatMul weights are sharded row-wise; the partial sums are
  reduced across ranks via in-graph AllReduce.
- The vocabulary projection is split along the vocabulary axis and the
  slices are gathered into rank 0, whose outputs the caller reads. A
  vocabulary the number of ranks does not divide stays whole on every rank.
- LayerNorm, RoPE, attention softmax, residual add, and embedding lookup
  are replicated on every rank.
- KV-cache shapes and head counts are adjusted so each rank only stores the
  KV heads its query heads use.

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
  MatMuls over constant-derived weights. Fused projections (one `qkv_proj`,
  one `gate_up_proj`) are supported.
- More ranks than KV heads only when the ranks divide by the KV heads and the
  query heads by the ranks (e.g. 2 KV heads and 32 query heads on 4 GPUs: two
  ranks share each KV head). Fewer ranks than KV heads work for any count.

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

The rank devices are part of the device name, as for `HETERO`:
`"TP:GPU.0,GPU.1,GPU.2,GPU.3"` runs one rank per listed device, in rank order
(the list is `ov::device::priorities`). At least two devices of one device type
are required; `compile_model` throws otherwise. Whether the devices can run
the collectives is the device plugin's answer, asked before anything is
compiled (GPU: Level Zero devices of one driver with peer access between every
pair, each listed once). Only GPU devices can host ranks today.

What the list is checked for, before anything is compiled:

| Input | Result |
|---|---|
| fewer than two entries (`TP`, `TP:`, `TP:GPU.0`, `TP.0`) | `TP requires at least 2 devices, got N ("<list>")` |
| an empty entry (`TP:GPU.0,,GPU.1`) | refused, naming the list |
| two device types (`TP:GPU.0,CPU`), another virtual device (`TP:HETERO:GPU.0,GPU.1`) | `all rank devices must have one device type` |
| a device type without collectives (`TP:CPU,CPU`, `TP:AUTO,AUTO`) | `the CPU plugin does not support tensor-parallel collectives` |
| a device type nobody registered (`TP:NPU.0,NPU.1` without an NPU plugin) | the core's `Device with "NPU" name is not registered` |
| a device that does not exist (`TP:GPU.0,GPU.6`, `GPU.-1`, `GPU.x`) | the device plugin's reason: `device GPU.6 of the group does not exist` |
| one device twice, also by another name (`TP:GPU.0,GPU.0`, `TP:GPU,GPU.0`) | the device plugin's reason: `device GPU.0 appears in the group more than once` |
| weights (and a fixed staging) that do not fit a device | see [Troubleshooting](#tp-the-model-does-not-fit-even-split-over-n-devices) |

Entries are trimmed of spaces; names are case-sensitive, like every OpenVINO
device name. `ov_internal_tp_comm_buffer_size` larger than a device allocates
at once is refused by name; below 512 bytes the GPU refuses it as too small.

| Property                   | Type       | Default | Description                                                                              |
|----------------------------|------------|---------|------------------------------------------------------------------------------------------|
| `TP_STRATEGY`              | string     | `COLUMN_ROW` | Partitioning strategy (Megatron-style column/row parallelism; the only value). |
| `COLLECTIVE_BACKEND`       | string     | `AUTO`  | Collective implementation of the rank devices' plugin, passed to it unchanged. GPU: `L0_P2P`; an unknown name is rejected with the valid ones. |
| `COMMUNICATION_TIMEOUT_MS` | `int`      | `5000`  | Longest a rank waits inside a collective before the group is aborted. `0` waits forever; a negative value is rejected at `compile_model`. |
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
a prompt's prefill, which is the hidden size times the prompt length, allocated when that prompt
first arrives. Setting it reserves the memory up front, before the KV cache is
sized, and caps it; a message that does not fit is sent in chunks. Every chunk
adds a little to the time to first token, more on more ranks. Decode
messages are small and never split. On more than two ranks a chunked prompt
is summed in a different order, so the outputs may differ from the unchunked
run in the last bits, the same way on every run.

The vocabulary projection (LM head) is split across the ranks too. When it shares its weight
with the embedding (tied weights), the embedding table stays whole on every rank and the split
adds a cut copy of the table, `table / N`, on each; `OV_TP_VERBOSE=LOG_INFO` (a build with
`-DENABLE_TP_DEBUG_CAPS=ON`) reports its size.
The environment variable `OV_TP_DISABLE_LM_HEAD_SHARDING=YES` keeps the LM head whole instead:
less device memory, a slower decode step. It is part of the model cache key.

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
collectives themselves do not appear there as a separate entry; their
measurements are the GPU plugin's `OV_GPU_TP_PROFILING` (a build with
`-DENABLE_DEBUG_CAPS=ON`), and the TP plugin's own timings are ITT tasks, see
[DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables).

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
- **Weightless caching works** (`ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE)`
  with a model read from IR). A rank weight that is a contiguous slice of a
  weight in the model's `.bin` is read back from there; the others (slices
  across rows) stay in the entry, so for a compressed model it holds a fraction of the weights.
- **Properties for the rank devices are part of the cache key**, whether given
  in the call (`ov::device::properties(...)`) or set on `TP`.
- `compiled.get_property(ov::loaded_from_cache)` tells you which path was
  taken.

## Paged Attention and Continuous Batching

The KV cache of a paged-attention model is split across the GPUs, so the
application does not allocate it. The compiled model offers
`ov::plugin_managed_cache` instead; OpenVINO GenAI's continuous-batching
pipeline uses it on its own. Everything it reports is about the model as if it
were not split:

```cpp
auto cache = compiled.get_property(ov::plugin_managed_cache);
auto group = cache.get_groups()[0];                   // KEY_VALUE: key_cache.N / value_cache.N
size_t blocks = cache.get_available_memory() / group.block_size_in_bytes;
cache.allocate(0, blocks, /*zero_initialize=*/false);  // bound to every infer request
// ... reallocate(), copy_blocks(), zero_blocks(), deallocate()
```

- The cache inputs cannot be set or read through the infer request.
- Inference before `allocate()` (or after `deallocate()`) fails.
- A failed `reallocate()` keeps the previous cache and its contents.

## Limitations

- **Architecture coverage.** The plugin recognizes attention and MLP blocks by
  their structure, then proves that the split computes what the model does. A
  block it recognizes only partly fails the compilation with a message naming
  the node (`[TP] Sharding verification failed at node ...`) instead of producing
  wrong output. A weight fed from a live parameter rather than a constant (a LoRA
  adapter) is not supported.
- **Quantized weights.** Weights of 8 bits and more (f32, f16, bf16, f8e4m3,
  f8e5m2, i8, u8) and 4-bit ones (u4, i4, nf4, f4e2m1) are split; a model with
  weights of 1, 2, 3 or 6 bits is rejected, naming the weight. A quantized
  weight is cut on quantization-group boundaries. When the split does not line
  up with them, the weight is cut at
  run time instead; the plugin warns and names the weight, and the device keeps
  it decompressed (see the warning below).
- **Single-process.** All ranks execute in the same process on plugin-owned
  threads. Multi-process / multi-host configurations are out of scope.
- **One TP request at a time per device.** TP requests of every TP model in
  the process that shares a device with another run one after another.
- **No linear-attention models.** Models with linear-attention state
  (hybrid models) are rejected at compile time.
- **No KV-cache eviction.** A paged-attention model compiled for cache
  eviction (GenAI `SchedulerConfig::use_cache_eviction`) fails the compilation
  (`The block around it is not fully recognized by any TP block pass`).
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

### `TP requires at least 2 devices, got 1`

The device name lists fewer than two rank devices (`"TP"` alone lists none). Name
them: `"TP:GPU.0,GPU.1"`; `core.get_available_devices()` shows which exist.

### `[TP] The model does not fit even split over N devices`

Before building any rank, the plugin compares each rank's weights with its
device's free memory (total minus what every context of the process uses) and
stops if they alone do not fit. The message gives the rank, both sizes, the part
of the weights every rank keeps whole (embedding, norms, layers no block pass
splits) and the fewest devices with that much free memory that would hold the
rest -- or that no number would, when the whole part alone is too big. The
weights are counted low (a cut weight by its share, f32 at two bytes per
element), so a model that passes can still run out of memory during compilation
or in the KV cache; one that fails cannot fit.

### Compilation fails on rank > 0 with an out-of-memory error

The weights fit, but what comes on top does not. The current sharding divides
the weights of the column- and row-parallel MatMuls and the vocabulary
projection across ranks, but does **not** shard the embedding table or
activation memory (a vocabulary projection that shares its weights with the
embedding adds its slice on top of the whole table; `OV_TP_DISABLE_LM_HEAD_SHARDING=YES` keeps it
whole). Effective memory per rank is
roughly `(sharded weights / N) + (unsharded weights) + (full activation
budget)`; `OV_TP_VERBOSE=LOG_INFO` prints each rank's weights next to its
device's free memory.

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
whole model moves at their pace. The percentage in the message is the share of
their time the other ranks spend waiting (one head out of the busier ranks'
heads); the warning only appears above 10%.

Choose a number of devices that divides the KV head count to make it go away.

### `[TP] Warning: N weights cannot be cut at the rank boundaries at compile time`

Not an error: the model runs correctly. A rank's boundary falls inside a quantization group of
the named weight, usually the attention output projection: a rank's query heads x `head_dim` is
not a whole number of groups (e.g. a `head_dim` of 64, an odd number of query heads per rank and
groups of 128). Such a weight is cut after
decompression, so the device keeps it in f16: about four times the memory of an int4 weight and
a markedly slower prefill for that layer.

Choose a number of devices that gives every rank whole groups, or compress the model with a
group size that divides a rank's share of the heads.

### `[TP] '<weight>' (<block>) splits ... unit(s) for N ranks: a rank would get none`

Every rank needs at least one unit of every split: a whole query group or KV head in attention,
a whole quantization group in the MLP. A model too small for the number of devices (an MLP of
two quantization groups on three devices, say) is rejected with the largest number of devices
that works. Use that many or fewer.

### `[TP] Model has H Q heads and K KV heads; world size N does not divide them evenly`

With more ranks than KV heads, every rank needs one whole KV head and an equal
share of its query heads, so N has to be a multiple of the KV head count and
divide the query head count. Use fewer devices, or a number that fits.

### `[TP] Sharding verification failed at node ...`

The plugin could not prove that the split model computes what the original
does: a block of the model is recognized only partly, or an operation on split
data has no sharding rule. The message names the node, its type, the input
shapes and states, and where each state came from. The model is not supported
as it is; please report it with the message.

### A collective times out or the group aborts

The error names the rank and the stage it gave up at. A rank that never
arrives usually means the surrounding application is doing something
uneven across ranks; device work that never completes usually means a
driver-level problem. Raise `COMMUNICATION_TIMEOUT_MS` for very large
payloads on slow interconnects. The compiled model is not reusable after an
abort.

For a trace of what each rank's collectives were doing, build with
`-DENABLE_DEBUG_CAPS=ON` and set `OV_GPU_TP_VERBOSE=LOG_TRACE`.

## Pointers

- Internal architecture and the device-side AllReduce pipeline:
  [ARCHITECTURE.md](ARCHITECTURE.md).
- Profiling, debugging recipes, environment variables, and how to add
  sharding patterns: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

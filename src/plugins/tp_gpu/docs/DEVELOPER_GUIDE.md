# Tensor Parallel Plugin — Developer Guide

## Build

The plugin is on by default in OpenVINO builds:

```bash
cd openvino/build
cmake -DENABLE_TP_GPU=ON ..    # ON by default
make -j$(nproc) openvino_tp_gpu_plugin
```

Produces `bin/intel64/Release/libopenvino_tp_gpu_plugin.so` and
registers `TP_GPU` in `plugins.xml`.

To disable:

```bash
cmake -DENABLE_TP_GPU=OFF ..
```

The plugin links against `openvino::runtime`, `openvino::runtime::dev`,
the L0 loader (`ze_loader`), an OpenCL ICD (used to compile the embedded
kernel), and the intel_gpu plugin's public headers (only for the
`tp_allreduce` op definition shared across the impls).

## Source Layout

See [ARCHITECTURE.md § Source Tree](ARCHITECTURE.md#source-tree-current)
for the current file layout. Key entry points:

| File | Role |
|---|---|
| `src/plugin.cpp` | `Plugin::compile_model` — config parse, shared L0 context creation, per-rank `GraphRewriter::rewrite` and per-rank intel_gpu compilation. |
| `src/graph_rewriter.cpp` | `analyze` (detects shardable MatMuls and model dimensions) and `rewrite` (per-rank weight slicing, KV-cache shape patching, TPAllReduce insertion). |
| `src/tp_device_coordinator.cpp` | All L0 work: per-rank queues/cmdlists, kernel build via OpenCL→native-binary, plan caching, execute_plan with timestamp profiling. |
| `src/infer_request.cpp` | `std::async` fan-out over ranks; collects rank 0 output. |
| `src/plugins/intel_gpu/src/graph/impls/ocl/tp_allreduce.cpp` | OCL primitive that calls into `TPDeviceCoordinator::allreduce`. |

## Tunables

They are plugin options, not bare environment reads: the table lives in
`include/tp_gpu/options.inl` and the X-macro there gives each entry a typed
getter, a default, validation, and environment and config-file support. An
option's environment variable is `OV_` plus its property key.

The debug tier needs `-DENABLE_TP_GPU_DEBUG_CAPS=ON` (implied by
`-DENABLE_DEBUG_CAPS=ON`). Without it those options have no getters at all,
so the guarded code folds away instead of costing a branch.

| Environment variable | Effect | When to enable |
|---|---|---|
| `OV_TP_VERBOSE=LOG_INFO` | One-line summaries per compile: sharding plan, shared context, per-rank op counts, which queue the collective rides. | First thing to reach for when something looks wrong. |
| `OV_TP_VERBOSE=LOG_TRACE` | Per-call `[TP][L0]` step-by-step tracing. | Only when chasing a hang or a correctness crash. |
| `OV_TP_PROFILING=HOST` | Host-side measurement: rendezvous phases, per-phase splits per rank, record breakdown, scratch arena. Costs about 0.6 ms a token and leaves the execution schedule alone. | The default choice when measuring. |
| `OV_TP_PROFILING=DEVICE` | Device-side kernel timestamps: memcpy and kernel durations, PCIe throughput. Needs timestamp event pools, which forbid the device-side event reset and move the group onto the single-threaded rank-0 schedule -- about 2.4 ms a token, and what it measures is **not** a production run. | Only when the device-side numbers are the question. |
| `OV_TP_PROFILING=ALL` | Both of the above. | |
| `OV_TP_DUMP_PERIOD=N` | Rank-0 collectives between two dumps. Unset means one dump per inference. | When one report per inference is too coarse or too noisy. |
| `OV_TP_USE_COPY_ENGINE=1` | Route the cross-device memcpy onto the dedicated copy ordinal (when the device exposes one). | Only for prefill-bound benchmarks. Net-negative for warm decode (extra `ExecuteCommandLists` per rank). |

Verbosity and profiling are deliberately separate: raising the verbosity to see
what a run is doing should not start timing it, and asking for timings should
not require guessing which log level carries them.

Profile output anatomy (one line per inference):

```
[TP][PROF] r0 calls=44 rebuilds=0  totals: ph1=… ph2=… (record=… exec=…) ph3=…  per-call: …
[TP][PROF]   exec breakdown: reset=…ms (…%) submit=…ms (…%) sync_r0=…ms (…%) sync_r1=…ms (…%)
[TP][PROF]   device steps (max across ranks): memcpy=…ms kernel=…ms (sum=…) ts_query_overhead=…ms
[TP][PROF]   throughput: bytes/call=…MB  per-dir(dev_copy)=… GB/s  full-duplex(dev_copy)=… GB/s  effective(exec)=… GB/s
```

Field semantics:

- **rebuilds** — number of full plan rebuilds since process start. After
  the first prefill of a given max-shape this should be 0.
- **record** vs **exec** — host time spent (re-)recording cmdlists vs
  submitting + syncing them.
- **sync_r0 / sync_r1** — first and second `zeCommandQueueSynchronize`
  in the regular path. `sync_r1` near zero means rank 1 finished before
  rank 0; otherwise rank 1 was the straggler.
- **memcpy / kernel** — max across the two ranks' device-side timings;
  this is the critical path on the device.
- **per-dir(dev_copy)** — `bytes_per_call / dev_copy_time` — the actual
  one-direction PCIe DMA bandwidth. Compare to PCIe practical sustained
  (~12–14 GB/s for PCIe 4.0 x8). The `full-duplex` figure is `2×` the
  per-direction value (both peers pump in opposite directions
  simultaneously).
- **effective(exec)** — `bytes_per_call / exec_per_call` — a
  user-visible figure including event-reset, submit, and sync overhead.

## Tests

Automated suites live under `tests/`, hand-driven binaries under
`samples/`.

### `ov_tp_gpu_unit_tests`

Two groups in one binary:

- Graph rewriter: structural analysis of a synthetic transformer block,
  per-rank sharding for world sizes 2/3/4, collective insertion, bias,
  KV-cache localization. Needs no GPU.
- `TPDeviceCoordinator`: AllReduce correctness on f16/f32, plan caching
  across shifting pointers and sizes, scratch growth, in-place operation,
  wider worlds. Skips when fewer than two discrete GPUs are present.

### `ov_tp_gpu_func_tests`

Accuracy of the whole pipeline: compiles the same synthetic block on a
single GPU and on TP_GPU, feeds both the same input and compares the
outputs. Parametrized over world size {2, 3, 4} and inference precision
{f16, f32}; the precision is pinned on both sides, because left alone the
GPU picks f16 and one ULP there is the same magnitude as a real sharding
error. Skips when the machine has too few GPUs.

Both register with CTest under the `TP_GPU` label:

```bash
ctest -L TP_GPU
```

### `tp_benchmark` — end-to-end comparator and prefill+decode driver

```bash
./bin/intel64/Release/tp_benchmark \
    --model /path/to/openvino_model.xml \
    --prefill-len 1024 \
    --gen-len 16
```

Flags:

- `--model` — path to a Llama-family OpenVINO IR.
- `--prefill-len N` — initial prompt length to fabricate (random ids).
- `--gen-len M` — number of decode steps to run after prefill, with a
  growing attention mask and advancing position ids; **no
  `state.reset()`** between decode steps, matching realistic LLM
  serving.
- `--device GPU` — override the baseline device used to compare against
  TP (defaults to `GPU.0`).

Output includes:
- The plain `[TP] Infer breakdown:` line emitted by `infer_request.cpp`.
- One `[TP][PROF]` block per `TP_PROF` window if the env var is set.
- Top-K logits comparison between TP and the single-GPU baseline.

## Level Zero Pitfalls

### Copies must be enqueued on the source device

`zeCommandListAppendMemoryCopy` has to go on the command list of the
device that **owns the source memory**. Enqueue it on the destination
device's list and the driver reports `ZE_RESULT_SUCCESS` while silently
transferring nothing. This is why every peer copy in
`tp_device_coordinator.cpp` is recorded onto the sending rank's list,
never the receiving one. A collective that returns stale data with no
error anywhere is almost always this rule being violated.

### One shared context spans every device

Cross-device copies and events only work when the USM allocations and
the event pool come from a single `ze_context` created over all
participating devices (`zeContextCreateEx`). The plugin builds it once
in `plugin.cpp` and hands it to every rank.

### Dedicated copy engines are optional

Discrete GPUs expose a copy-only command queue group (Link Copy Engine)
that lets transfers overlap with kernel work; integrated GPUs usually do
not. Code that selects a copy ordinal must fall back to the compute
ordinal rather than assume the group exists — see `select_copy_ordinal`.

### Topology cost

A gather/scatter through one coordinating rank costs `2*(N-1)*bytes`
over that rank's link, so the coordinator becomes the bottleneck from
N >= 3. The ring reduce-scatter + all-gather the plugin uses costs
`2*bytes` per link independently of N, which is why it is the default
above two ranks.

## Debugging Recipes

### "Plan rebuilds every call"

Symptom: `[TP][PROF]` reports `rebuilds=N/N` even on warm decode.

Causes and fixes:
- Input/output USM pointers shift every call (intel_gpu re-allocates).
  Investigation: log `slot->in_ptrs` vs `rdz.in_ptrs` in
  `allreduce()`'s rebuild branch. Look for an intel_gpu memory pool that
  isn't reusing buffers; usually fixed by pinning the model output to a
  long-lived USM tensor on the user side.
- Sequence length shrinks below the plan's recorded `n`. This is
  expected on the first decode step and is now amortized by the
  `n_capacity` reuse: ensure `Plan::n_capacity` is set in `build_plan`
  and checked in `allreduce()`'s rebuild predicate. If you broke
  capacity logic, expect ~16 ms of wasted record cost per inference.

### "Hang on the second inference"

Almost always traces back to issuing `zeMemFree` / `zeCommandListReset`
on resources still in flight from the previous submission. The
coordinator currently guards against this in three places:

- `destroy_plan` syncs every rank's compute_queue (and copy_queue when
  set) before calling `zeMemFree`/`zeEventDestroy`.
- `record_rank` syncs queues before `zeCommandListReset`.
- `~TPDeviceCoordinator` indirectly relies on the same syncs via
  `destroy_plan`.

If you add a new code path that frees or resets L0 resources, place
queue syncs at the top of it.

### "AllReduce produces zeros / NaNs"

Most common root causes (in order of frequency):

1. **Memcpy enqueued on the wrong rank's cmdlist.** L0 silently drops
   transfers when the source device is not the one issuing the copy
   command. Source-side enqueue is mandatory; verify in `record_ring_rank`
   or `record_pair_rank` that the memcpy lives on `self.copy_list` or
   `self.compute_list`, not on the peer's.
2. **Cross-device wait on a non-timestamp pool.** The Intel L0 driver
   has shipped versions where event-wait across devices only resolves
   reliably when the source pool was created with
   `ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP`. The coordinator sets this flag
   unconditionally; do not remove it as part of "cleanup".
3. **Reusing an event without `zeEventHostReset`.** Each event is
   single-shot. `execute_plan` resets every event at the start of the
   call.

### "Throughput well below PCIe practical peak"

After confirming `bytes/call` is large enough (≥ 4 MB), look at:

- `submit` percentage in the breakdown: if it's high (>30%), small
  transfers are dominated by host API cost — try enabling
  `TP_COPY_ENGINE=1` for prefill, or skip TP for that workload.
- `sync_r1` non-trivial: rank 1 is straggling. Check whether something
  on rank 1's device is contending with the reduce kernel (e.g. another
  process, or a stray GPU.1 inference request issued from the
  surrounding application).
- `kernel` time dominating: the reduce kernel in `allreduce_sum.cl` is
  a reference scalar-load implementation. Vectorizing to `half8`/`float8`
  loads would shave ~30% off the kernel time but is a small share of
  the wall clock; only worth it once memcpy is at peak.

## Adding a New Sharding Pattern

The current rewriter is name-driven. To add support for an architecture
that doesn't expose `layers.{N}.self_attn.q_proj` etc., extend
`graph_rewriter.cpp::analyze`:

1. Add additional `name.find(...)` branches that map a MatMul friendly
   name to a `ShardingPlan::LinearDesc` with `layer_idx`, `role`, and
   `is_column_parallel`.
2. If the model uses different shape inference around QKV (different
   reshape factor, no GQA, fused QKV, …), update the shape-constant
   patcher in `rewrite()` accordingly. In particular:
   - `local_q_heads` / `local_kv_heads` are derived assuming a
     `[B, S, num_heads, head_dim]` reshape after Q. Custom reshapes need
     custom predicates.
   - The KV-cache shape constant patch (search for `kv_init_patched`)
     looks for `Constant → Concat → Broadcast → ReadValue` — different
     state-init topologies need their own pattern.
3. Add a case to `tests/unit/graph_rewriter_test.cpp`, extending
   `BlockConfig` in `tests/common/tp_test_models.hpp` when the new
   pattern needs a different synthetic block.

## Adding a New TP Op

The plugin defines exactly two ops, `TPAllReduce` and `TPGather`, and
both are emitted by the rewriter. To add a third:

1. Header in `include/tp_gpu/op/`, source in `src/op/`.
2. Inherit `ov::op::Op`, declare with
   `OPENVINO_OP("TPMyOp", "tp_gpu", Op)`.
3. Implement constructors, `validate_and_infer_types`,
   `clone_with_new_inputs`, `visit_attributes`. The two existing ops
   are good templates.
4. Add an OCL primitive in
   `src/plugins/intel_gpu/src/graph/impls/ocl/`. Keep the primitive POD:
   carry `group_id` / `collective_id` / `rank` as attributes and resolve
   the coordinator at execution time via
   `instance.get_network().get_collective_comm_registry()->get_group(group_id)`.
5. Register in `tp_*_impls.cpp` (priority OCL_static, OCL_dynamic, then
   any CPU fallback).

## Layout Conventions

- All public headers live under `include/tp_gpu/` and are
  consumed by both the plugin and the intel_gpu impls.
- Internal headers (e.g. `compiled_model.hpp`, `plugin.hpp`,
  `graph_rewriter.hpp`) stay in `src/`.
- The shared L0 context type is internal — do not promote it to a
  public header. The intel_gpu impl reaches the `TPDeviceCoordinator`
  through the network's `CollectiveCommRegistry`, never the L0 context
  directly.

## Where Performance Lives

If you are looking to improve perf, the work has settled into these
areas (in order of remaining headroom):

1. **Async/overlapped AllReduce** — currently every collective is a
   synchronous device round-trip from rank 0's thread, blocking the
   surrounding graph. Decoupling submit from sync (e.g. submit from rank
   0's worker thread, sync at the next tp barrier) can hide most of
   the 80–90 µs/call host overhead behind compute. Largest single
   remaining win on small-shape workloads.
2. **Kernel vectorization** — `allreduce_sum.cl` reads/writes scalars.
   Switching to `half8`/`float8` reduces device-side time. Gains only
   matter once host overhead is hidden.
3. **Sequence Parallel** — replace AllReduce with
   ReduceScatter+AllGather around the layer norms, parallelizing the
   replicated norms as well. Significant rewrite in `graph_rewriter`,
   meaningful (5–10%) win on bigger models.

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
| `OV_TP_PROFILING=DEVICE` | Device-side numbers only: how long the collective owns the model queue, bytes moved, effective bandwidth, and what share of device time the collectives take. Runs the production schedule; the kernel-timestamp query costs about 3 us a collective, so host figures from such a run are not comparable with a HOST run -- which is why it does not print them. | Only when the device-side numbers are the question. |
| `OV_TP_PROFILING=ALL` | Both of the above. | |
| `OV_TP_DUMP_PERIOD=N` | Rank-0 collectives between two dumps. Unset means one dump per inference. | When one report per inference is too coarse or too noisy. |

Verbosity and profiling are deliberately separate: raising the verbosity to see
what a run is doing should not start timing it, and asking for timings should
not require guessing which log level carries them.

Profile output anatomy (one block per dump period, one dump per inference by
default):

```
[TP][RANK|host] calls=…  arrival spread=…us/call
[TP][RANK|host]   rank N: late=…us arrived_last=…%  ph1=…us ph2=…us  segment mean=…us min=…us max=…us
[TP][RANK|host]     ph2 split: gate=…us record=…us(Nx) evt_wait=…us evt_reset=…us(Nx, blocked Nx) append=…us rest=…us
[TP][RANK|dev]    rank N: collective holds the queue for …us max=…us (N samples)
[TP][TOTAL|host] r0 calls=…  rebuilds=… records=…  totals: … per-call: …
[TP][TOTAL|host]   schedule of N recordings: pair=… ring=… halving=…
[TP][TOTAL|host]   record breakdown: records=… per-record=…ms (drain=… reset=… append=… close=…)
[TP][TOTAL|host]   scratch: payload_capacity=…MB total=…MB generation=… grows=… allocations=…
```

The tags say what the numbers are about and where they came from.
`RANK` is per rank and exists to expose imbalance between them; `TOTAL` is
rank 0's aggregate and exists to price the collective as a whole. `host` means
a host clock, `dev` means GPU kernel timestamps -- the `dev` lines appear only
under `TP_PROFILING=DEVICE` or `ALL`.

Field semantics:

- **ph1** — the enter barrier: waiting for the other ranks to arrive.
- **ph2** — everything from leaving the barrier to having handed the work over.
- **late / arrived_last / arrival spread** — how far behind the first arrival
  this rank was, and how often it was the last one in. The group moves at the
  speed of the last rank, 65 times per token.
- **segment** — model work between leaving one collective and reaching the
  next. This is where the arrival skew is built, which is why its extremes are
  kept rather than just the mean.
- **gate** — waiting for rank 0 to settle the signature and the staging arena.
  On rank 0 itself it is the time spent settling them.
- **record** — laying down this rank's command lists, with the count in
  brackets. Rare, but every miss lands in time to first token.
- **evt_wait / blocked** — waiting for the previous splice of this same list to
  finish. `blocked` counts how often that wait actually had to block; anything
  but zero means the host has caught up with the device.
- **evt_reset** — clearing the completion event. Under `DEVICE` this also
  carries the kernel-timestamp query, which is the one measurable cost device
  profiling adds.
- **append** — the splice call itself.
- **rest** — ph2 minus everything above, i.e. what is not accounted for.
- **holds the queue for** — how long the collective occupied the rank's model
  queue, read off the completion event. This is the one part of the cost no
  host phase contains: the splice hands the work over and returns, so the
  copies, the reduce kernel and the waits on peers all happen after every host
  measurement has ended.
- **rebuilds** — staging arena growths since process start. After the first
  prefill of a given max-shape this should stop moving.
- **schedule** — which of the three schedules the recordings chose. `halving`
  is default-on and bounded by payload, so the split says whether that bound is
  anywhere near right.
- **record breakdown** — divided by the number of recordings, not calls:
  `drain` is the queue drain before a reset, `append` the command building,
  `close` the `zeCommandListClose`.

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

- `destroy_plan` syncs every rank's compute_queue before calling
  `zeMemFree`/`zeEventDestroy`.
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
   carry `collective_id` / `rank` as attributes and resolve
   the coordinator at execution time via
   `instance.get_network().get_collective_comm_registry()->coordinator()`.
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

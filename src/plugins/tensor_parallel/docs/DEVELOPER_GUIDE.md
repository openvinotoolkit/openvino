# Tensor Parallel Plugin — Developer Guide

## Build

The plugin is on by default in OpenVINO builds:

```bash
cd openvino/build
cmake -DENABLE_TENSOR_PARALLEL=ON ..    # ON by default
make -j$(nproc) openvino_tensor_parallel_plugin
```

Produces `bin/intel64/Release/libopenvino_tensor_parallel_plugin.so` and
registers `TENSOR_PARALLEL` in `plugins.xml`.

To disable:

```bash
cmake -DENABLE_TENSOR_PARALLEL=OFF ..
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

## Tunables (all opt-in via env var)

| Env var | Effect | When to enable |
|---|---|---|
| `TP_PROF=N` | After every Nth rank-0 call, print aggregated profile: rendezvous phases, exec breakdown (reset / submit / sync_r0 / sync_r1), device-side memcpy & kernel time from `zeEventQueryKernelTimestamp`, and PCIe throughput. | Always when measuring. Pick `N = 2 × num_layers` to get one report per inference. |
| `TP_DBG=1` | Extremely verbose `[TP][L0]` step-by-step tracing inside `execute_plan` and `record_plan`. | Only when chasing a hang or correctness crash. |
| `TP_COPY_ENGINE=1` | Route the cross-device memcpy onto the dedicated copy ordinal (when the device exposes one). | Only for prefill-bound benchmarks. Net-negative for warm decode (extra `ExecuteCommandLists` per rank). |
| `TP_USE_IMMEDIATE=1` | Use immediate cmdlists + counter-based events for host-sync. | Currently does **not** work end-to-end on the validated driver — fix in progress. Do not enable for production. |

Profile output anatomy (one line per inference at `TP_PROF=44` for a
22-layer Llama):

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

Two test binaries live under `tests/`.

### `tp_test` — end-to-end comparator and prefill+decode driver

```bash
cd build_release
./bin/intel64/Release/tp_test \
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

### `tp_coordinator_test` — focused unit tests for `TPDeviceCoordinator`

Builds a synthetic two-rank coordinator with the shared L0 context and
exercises:
- AllReduce correctness on f16/f32, varying `n` from 1 element up to a
  few MiB.
- Plan caching (rebuild vs re-record vs reuse) across a sequence of
  calls with shifting pointers and shrinking `n`.
- Lifetime: repeated full destroy/build cycles to verify no L0 handles
  leak across plan churn.

The `samples/tp_allreduce_l0.cpp` binary is **not** a test — it is a
standalone L0 reproducer used to measure raw PCIe peer-copy bandwidth
without OpenVINO in the loop.

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
- `record_plan` syncs queues before `zeCommandListReset`.
- `~TPDeviceCoordinator` indirectly relies on the same syncs via
  `destroy_plan`.

If you add a new code path that frees or resets L0 resources, place
queue syncs at the top of it.

### "AllReduce produces zeros / NaNs"

Most common root causes (in order of frequency):

1. **Memcpy enqueued on the wrong rank's cmdlist.** L0 silently drops
   transfers when the source device is not the one issuing the copy
   command. Source-side enqueue is mandatory; verify in `record_plan`
   that the memcpy lives on `self.copy_list` or `self.compute_list`,
   not on the peer's.
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
3. Add a focused test in `tp_test.cpp` or, preferably, a small synthetic
   IR in `tests/` that exercises the new pattern.

## Adding a New TP Op

Mostly historical at this point — `TPReduceScatter`, `TPAllGather`,
`TPBroadcast` exist for future Sequence Parallel work, but none of them
are emitted by the current rewriter. If you wire one up:

1. Header in `include/tensor_parallel/op/`, source in `src/op/`.
2. Inherit `ov::op::Op`, declare with
   `OPENVINO_OP("TPMyOp", "tp_internal_opset", Op)`.
3. Implement constructors, `validate_and_infer_types`,
   `clone_with_new_inputs`, `visit_attributes`. The four existing ops
   are good templates.
4. Add an OCL primitive in
   `src/plugins/intel_gpu/src/graph/impls/ocl/` that retrieves
   `TPCoordination` from rt_info and dispatches to the appropriate
   coordinator method.
5. Register in `tp_*_impls.cpp` (priority OCL_static, OCL_dynamic, then
   any CPU fallback).

## Layout Conventions

- All public headers live under `include/tensor_parallel/` and are
  consumed by both the plugin and the intel_gpu impls.
- Internal headers (e.g. `compiled_model.hpp`, `plugin.hpp`,
  `graph_rewriter.hpp`) stay in `src/`.
- The shared L0 context type is internal — do not promote it to a
  public header. The intel_gpu impl reaches the `TPDeviceCoordinator`
  through `TPCoordination`, never the L0 context directly.

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

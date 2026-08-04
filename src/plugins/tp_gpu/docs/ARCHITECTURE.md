# Tensor Parallel Plugin — Architecture

## Overview

The Tensor Parallel (`TP_GPU`) plugin is an OpenVINO meta-plugin that
runs a single transformer model across multiple Intel discrete GPUs using
**Megatron-style tensor parallelism**: weights of large MatMuls are sharded
across ranks, partial sums are summed back via in-graph AllReduce
collectives, and the cross-device synchronization is implemented directly on
device memory through a shared Level Zero context.

```
┌──────────────────────────────────────────────────────────────┐
│                     User Application                         │
│   core.compile_model(model, "TP_GPU", config)       │
│   request = compiled.create_infer_request()                  │
│   request.infer()                                            │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│                  TP Meta-Plugin (Plugin)                     │
│                                                              │
│  compile_model():                                            │
│    1. Parse config (TP degree, device list).                 │
│    2. Build a single L0 context spanning all rank GPUs.      │
│    3. Analyze the model: detect Q/K/V/O projections and MLP  │
│       gate/up/down — produces a ShardingPlan.                │
│    4. For each rank: clone the model, slice weight Constants │
│       column- or row-parallel, patch num_kv_heads constants  │
│       and reshape shapes, insert TPAllReduce after each      │
│       row-parallel MatMul, validate the cloned graph.        │
│    5. Compile each per-rank model on its GPU through the     │
│       intel_gpu plugin under the shared L0 context.          │
│    6. Construct TPCoordination + TPDeviceCoordinator and     │
│       attach them to all ranks via rt_info on TPAllReduce.   │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│              CompiledModel (ICompiledModel)                  │
│                                                              │
│  Stores:                                                     │
│    - rank_compiled[N]    (per-rank ICompiledModel)           │
│    - shared coordination handles (kept alive by Plugin)      │
│                                                              │
│  create_infer_request() → InferRequest                       │
└──────────────┬───────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────┐
│              InferRequest (ISyncInferRequest)                │
│                                                              │
│  Stores: m_rank_requests[N]  (one inference request per GPU) │
│                                                              │
│  infer():                                                    │
│    1. Replicate user inputs across ranks.                    │
│    2. Launch each rank in its own std::async thread.         │
│    3. Each rank's graph executes locally; whenever it hits a │
│       TPAllReduce node, the OCL impl in intel_gpu reaches    │
│       through the rt_info-bound TPCoordination into the      │
│       TPDeviceCoordinator and performs a synchronous,        │
│       in-place cross-device sum on USM-device memory.        │
│    4. Both ranks return; rank 0's outputs are exposed to     │
│       the user.                                              │
└──────────────────────────────────────────────────────────────┘
```

## Sharding Strategy (Megatron-style)

For a standard Llama-family decoder block with hidden size `H`,
`num_heads` query heads, `num_kv_heads` (GQA), and intermediate size `I`,
the rewriter shards the following MatMul weights:

| MatMul        | Parallel kind     | Weight slice axis | Shard size                         | Notes |
|---------------|-------------------|-------------------|------------------------------------|-------|
| `q_proj`      | column-parallel   | output (axis 0)   | `H × (num_heads/N · head_dim)`     | each rank computes `num_heads/N` query heads |
| `k_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  | requires `num_kv_heads % N == 0` |
| `v_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  | |
| `o_proj`      | row-parallel      | input (axis 1)    | `(num_heads/N · head_dim) × H`     | output → **AllReduce(sum)** |
| `gate_proj`   | column-parallel   | output (axis 0)   | `H × I/N`                          | |
| `up_proj`     | column-parallel   | output (axis 0)   | `H × I/N`                          | |
| `down_proj`   | row-parallel      | input (axis 1)    | `I/N × H`                          | output → **AllReduce(sum)** |

Operations that remain replicated on every rank (cheap relative to MatMul):
LayerNorm/RMSNorm, RoPE, attention softmax, residual add, SiLU, embedding
lookup, lm_head. SDPA and the KV-cache are **not** explicitly parallelized —
they fall out for free because Q/K/V are already head-sharded and the cache
shape constants get patched to `local_kv_heads`.

**Collectives per inference**: `2 × num_layers` AllReduce calls — one after
attention `o_proj`, one after MLP `down_proj`. This is the lower bound for
Megatron-style TP and cannot be reduced without changing the parallelism
scheme (e.g. Sequence Parallel / SP+TP).

## Data Flow Inside a Decoder Layer

```
  X (replicated on rank 0 and rank 1)
   │
   ├──────────────────────────────┬──────────────────────────────
   ▼                              ▼
 LayerNorm  (replicated)        LayerNorm  (replicated)
   │                              │
 q,k,v_proj column-shard        q,k,v_proj column-shard
   │  (local heads only)          │  (local heads only)
 RoPE / SDPA / KV-cache         RoPE / SDPA / KV-cache
   │                              │
 o_proj row-shard               o_proj row-shard
   │  partial sum (rank 0)        │  partial sum (rank 1)
   ▼                              ▼
   └──────  TPAllReduce (sum) on device USM ───────┐
            via TPDeviceCoordinator               │
   ┌──────────────────────────────────────────────┘
   ▼                              ▼
 + residual                     + residual
 LayerNorm                      LayerNorm
   │                              │
 gate, up_proj column-shard     gate, up_proj column-shard
 SiLU * up                      SiLU * up
 down_proj row-shard            down_proj row-shard
   │  partial sum                 │  partial sum
   ▼                              ▼
   └──────  TPAllReduce (sum) on device USM ───────┘
   │                              │
 + residual                     + residual
   ▼                              ▼
 (next layer)                   (next layer)
```

## Component Map

```
Plugin
  └── TPL0SharedContext              (shared ze_context_handle_t across N GPUs)
  └── TPCoordination                 (one per CompiledModel; pinned in rt_info)
        └── TPDeviceCoordinator      (owns per-rank L0 queues, cmdlists, modules)
              └── Plan[]             (one per collective_id, cached & reused)
        └── SimpleBarrier            (legacy CPU AllReduce path; unused with DC)
  └── CompiledModel
        └── rank_compiled[N]          (intel_gpu per-rank ICompiledModel)
  └── InferRequest
        └── m_rank_requests[N]        (per-rank IAsyncInferRequest)
                                      ↓ during infer()
                                  intel_gpu OCL tp_allreduce_impl
                                      ↓
                                TPDeviceCoordinator::allreduce(...)
```

## Source Tree (current)

```
src/plugins/tp_gpu/
├── CMakeLists.txt
├── docs/                                  ← this directory
├── include/tp_gpu/
│   ├── op/                                ← public op headers (registered in
│   │                                        a private "tp_gpu")
│   ├── tp_coordination.hpp                ← TPCoordination (rendezvous + DC slot)
│   └── tp_device_coordinator.hpp          ← TPDeviceCoordinator public API
├── samples/
│   ├── allreduce_sum.cl                   ← reduce kernel source
│   ├── tp_allreduce_l0.cpp                ← standalone L0 reproducer / BW test
│   └── embed_spv.cmake                    ← bakes the .cl into the plugin .so
├── src/
│   ├── op/                                ← TPAllReduce, TPReduceScatter,
│   │                                        TPAllGather, TPBroadcast op classes
│   ├── plugin.{hpp,cpp}                   ← IPlugin entry point; builds
│   │                                        shared L0 context + DC
│   ├── compiled_model.{hpp,cpp}           ← ICompiledModel
│   ├── infer_request.{hpp,cpp}            ← ISyncInferRequest, std::async fan-out
│   ├── graph_rewriter.{hpp,cpp}           ← analyze + per-rank rewrite
│   ├── tp_l0_shared_context.hpp           ← RAII for the shared ze_context
│   ├── tp_device_coordinator.cpp          ← L0 plans, cmdlists, kernels, exec
│   ├── tp_embedded_kernels.h              ← generated from allreduce_sum.cl
│   └── version.cpp                        ← OV_DEFINE_PLUGIN_CREATE_FUNCTION
└── tests/
    ├── tp_test.cpp                        ← prefill+decode comparator vs GPU
    └── tp_coordinator_test.cpp            ← unit tests for TPDeviceCoordinator
```

## TPDeviceCoordinator (Device-side AllReduce)

The coordinator owns per-rank L0 resources and executes the cross-device
sum on USM-device memory. Each `allreduce()` call on rank R only triggers
real work on rank 0's thread; non-zero ranks rendezvous and wait.

### Per-rank state

For every rank, the coordinator creates:
- A compute queue + recordable command list on the compute ordinal
  (regular path), or an immediate command list (immediate path, gated by
  `TP_USE_IMMEDIATE`).
- Optionally a copy queue + command list on a copy-only ordinal when
  `TP_COPY_ENGINE=1` is set (off by default — see "Tunables" below).
- An `allreduce_sum_f16` and `allreduce_sum_f32` kernel compiled from
  `samples/allreduce_sum.cl`. The kernel module is built via OpenCL
  (`clBuildProgram`) on the device matched by UUID, then re-loaded into
  the shared L0 context as `ZE_MODULE_FORMAT_NATIVE`. This works around
  the lack of OCLC support in the L0 driver on the validated stack.
- For the immediate path, a per-rank counter-based event
  (`zexCounterBasedEventCreate2`, Intel L0 extension) used as a host-sync
  target without `zeCommandQueueSynchronize`.

### Plan caching

Each AllReduce site (`collective_id`) has a cached `Plan` that owns:
- A KERNEL_TIMESTAMP event pool with `ev_recv[2]` (signaled by each
  rank's memcpy) plus optionally `ev_ts_kernel[2]` (kernel-end probes,
  only created when `TP_PROF` is set).
- One `incoming` USM-device staging buffer per rank, sized to the
  largest seen `n` (`n_capacity`).
- The recorded command lists themselves.

The plan is reused as much as possible:
| Case                              | Action       | Cost per call |
|-----------------------------------|--------------|---------------|
| Same `(in_ptr, out_ptr, n, dtype)`| Plan reuse   | only event reset + submit + sync |
| Different pointers, `n ≤ capacity`| Re-record    | ~0.2 ms (no realloc) |
| `n > capacity` or different dtype | Full rebuild | ~0.4 ms (free + alloc + pool rebuild) |

`n_capacity` grows monotonically in a process, so a one-shot pre-warm
inference at the maximum expected sequence length amortizes all rebuild
cost; subsequent prefill/decode shapes only re-record.

### N=2 execute path (regular cmdlists)

```
rank 0                              rank 1
  |                                   |
  | host_reset(ev_recv,               | host_reset(ev_recv,
  |   ev_ts_kernel)                   |   ev_ts_kernel)
  |                                   |
  | submit copy_list                  | submit copy_list
  |   (memcpy in→peer.staging,        |   (memcpy in→peer.staging,
  |    signals ev_recv[0])            |    signals ev_recv[1])
  | submit compute_list               | submit compute_list
  |   wait ev_recv[1]                 |   wait ev_recv[0]
  |   reduce(out, in, staging)        |   reduce(out, in, staging)
  |                                   |
  | (host) sync compute_queue ───────►| (host barrier)
  |                                   |
  └── return ─────────────────────────┘
```

When no dedicated copy engine is configured, the memcpy and reduce live
on the same compute command list. Source-side enqueue is mandatory — a
peer-pull memcpy silently transfers zeros on the validated driver.

### N=2 execute path (immediate cmdlists, opt-in)

When `TP_USE_IMMEDIATE=1`:
- Each rank's commands are appended to its immediate cmdlist, which
  starts executing as soon as commands are enqueued.
- The reduce kernel signals a counter-based event (`cb_event_done`) at
  completion; the host syncs with `zeEventHostSynchronize(cb)`.
- `zeCommandListReset` cannot be issued on immediate cmdlists, so plan
  re-record uses a different path (commands are appended fresh per
  call). **This path is currently disabled by default** while a
  cross-device wait deadlock on the validated driver is being fixed.

### N>2 funnel path (legacy)

For world sizes greater than 2, the original funnel topology is retained:
- Workers (ranks 1..N-1) push their `in` to per-worker staging buffers
  on rank 0's device.
- Rank 0 waits on all `ev_recv[w]`, then folds the staging buffers into
  the running output via successive reduce kernel launches, signaling
  `ev_reduce` on the last fold.
- Rank 0 scatters the result back to each worker's `out_ptr`.

For TP=2 (the validated configuration) the symmetric path above is used
instead — it removes one round trip per call.

## intel_gpu Integration

The TP plugin contributes a single OCL primitive
(`src/plugins/intel_gpu/src/graph/impls/ocl/tp_allreduce.cpp`) that
implements the in-graph `TPAllReduce` op. This impl:
- Has `is_cpu() == false`, which forces intel_gpu to allocate inputs and
  outputs in `usm_device`. This is required for cross-PCIe peer copies.
- During `execute()`, retrieves `TPCoordination` from the op's rt_info,
  unwraps the device coordinator, and calls
  `dc->allreduce(collective_id, rank, in, out, n, dtype)` synchronously.

Registration order (see `tp_allreduce_impls.cpp`): `OCL_static`,
`OCL_dynamic`, `CPU_static`, `CPU_dynamic`. The CPU paths exist as a
fallback when no shared L0 context is available.

## Lifetime and Threading

- `TPL0SharedContext` is owned by the `Plugin` instance and outlives all
  per-rank `CompiledModel`s. `~TPL0SharedContext` calls `zeContextDestroy`.
- `TPDeviceCoordinator` is shared via `std::shared_ptr` between the
  `Plugin`, the `TPCoordination`, and through rt_info every TPAllReduce
  node — and therefore every per-rank intel_gpu graph.
- `infer_request.cpp` launches one `std::async(std::launch::async)` per
  rank; the implicit thread pool of std::async is sufficient for the 2-
  to 4-GPU configurations that have been validated.
- Inside `TPDeviceCoordinator::allreduce`, ranks use a 3-phase
  rendezvous (enter / execute / exit) over a `std::condition_variable`
  with generation counters — only rank 0 records and submits, the
  others wait.

## Tunables

| Env var             | Default | Effect |
|---------------------|---------|--------|
| `TP_USE_IMMEDIATE`  | `0`     | Immediate cmdlists + counter events. Currently broken on the validated driver; do not enable unless you are debugging it. |
| `TP_COPY_ENGINE`    | unset   | Route the cross-device memcpy onto the dedicated copy ordinal. Net negative on small (decode) transfers due to extra `ExecuteCommandLists` overhead per rank; useful only for prefill-bound benchmarks. |
| `TP_PROF`           | unset   | If set to a positive integer N, every Nth call from rank 0 prints aggregated host-side and device-side timings (rendezvous phases, exec breakdown, kernel-timestamp memcpy/kernel durations, PCIe throughput). |
| `TP_DBG`            | unset   | Verbose tracing of every L0 step taken by `execute_plan`. Very noisy; for crash investigation only. |

## Validated Configurations

- **Hardware**: 2× Intel Arc B580 (BMG) on PCIe 4.0 x8 each.
- **Software**: Intel L0 driver (compute-runtime) with the
  `zexCounterBasedEventCreate2` extension available.
- **Models**: Llama-architecture decoders with GQA (e.g. TinyLlama-1.1B,
  Llama-3.2-1B/3B, Qwen2-7B). Models are required to expose
  `layers.{N}.{self_attn|mlp}.{qkv,o,gate,up,down}_proj` MatMul names —
  the rewriter is name-driven.
- **TP degree**: 2 (production path). N>2 builds and runs but uses the
  legacy funnel topology.

## Known Limitations

- The graph rewriter recognizes Llama-style layer naming via regex.
  Other architectures (Mistral, MPT, Falcon, GPT-J, …) need either a
  matching naming convention or explicit pattern additions.
- Dynamic quantization (`hint::dynamic_quantization_group_size > 0`) is
  force-disabled by `Plugin::compile_model`. The oneDNN i8×i4 BRGEMM
  kernel produces shape-dependent results that diverge between
  unsharded and sharded MatMuls at sequence lengths ≥ 80.
- Model import/export and remote contexts from the user are not
  supported. The plugin always creates its own shared L0 context.
- Stateful inference works (KV-cache constants are patched), but
  `state.reset()` mid-session destroys plan capacity tracking on the
  next forward — only an issue for benchmarks that toggle reset.

## Pointers

- Detailed correctness/perf debugging steps and tunables: see
  [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
- External API and configuration: see [USER_GUIDE.md](USER_GUIDE.md).

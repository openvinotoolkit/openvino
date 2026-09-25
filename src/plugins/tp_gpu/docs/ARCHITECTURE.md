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
│   core.compile_model(model, "TP_GPU", config)                │
│   request = compiled.create_infer_request()                  │
│   request.infer()                                            │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         TP Plugin                            │
│                                                              │
│  compile_model():                                            │
│    1. Parse config (TP degree, device list).                 │
│    2. Build a single L0 context spanning all rank GPUs.      │
│    3. Analyze the model: walk back from every attention op   │
│       to find the Q/K/V/O projections and the MLP            │
│       gate/up/down — produces a ShardingPlan.                │
│    4. For each rank: clone the model, slice weight Constants │
│       column- or row-parallel, patch num_kv_heads constants  │
│       and reshape shapes, insert TPAllReduce after each      │
│       row-parallel MatMul and a TPGather after the sharded   │
│       vocabulary projection, validate the cloned graph.      │
│    5. Compile each per-rank model on its GPU through the     │
│       intel_gpu plugin under the shared L0 context.          │
│    6. Construct the TPDeviceCoordinator, publish it in a     │
│       CollectiveCommRegistry and inject that registry into   │
│       every rank compiled model via set_property().          │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              CompiledModel (ICompiledModel)                  │
│                                                              │
│  Owns everything the ranks share:                            │
│    - rank_compiled[N]     per-rank intel_gpu model           │
│    - TPL0SharedContext    ze_context over all devices        │
│    - TPDeviceCoordinator  queues, recordings, staging        │
│    - RankWorkers          a thread per non-zero rank         │
│    - CacheController      paged-attention cache              │
│    - inference mutex      one outer inference at a time      │
│                                                              │
│  Built by the plugin, then moved in and kept alive here.     │
│                                                              │
│  create_infer_request() → InferRequest                       │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              InferRequest (ISyncInferRequest)                │
│                                                              │
│  Stores: m_rank_requests[N]  (one inference request per GPU) │
│                                                              │
│  infer():                                                    │
│    1. Replicate user inputs across ranks.                    │
│    2. Hand ranks 1..N-1 to persistent worker threads; rank 0 │
│       runs on the calling thread.                            │
│    3. Each rank's graph executes locally; whenever it hits a │
│       TPAllReduce or TPGather node, the OCL impl in          │
│       intel_gpu resolves the coordinator from the network's  │
│       registry and hands the recorded collective to the      │
│       model's own command queue.                             │
│    4. All ranks return; rank 0's outputs are exposed to the  │
│       user.                                                  │
└──────────────────────────────────────────────────────────────┘
```

## Sharding Strategy (Megatron-style)

For a standard Llama-family decoder block with hidden size `H`,
`num_heads` query heads, `num_kv_heads` (GQA), and intermediate size `I`,
the rewriter shards the following MatMul weights:

| MatMul        | Parallel kind     | Weight slice axis | Shard size                         | Notes                                        |
|---------------|-------------------|-------------------|------------------------------------|----------------------------------------------|
| `q_proj`      | column-parallel   | output (axis 0)   | `H × (num_heads/N · head_dim)`     | each rank computes `num_heads/N` query heads |
| `k_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  | requires `num_kv_heads % N == 0`             |
| `v_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  |                                              |
| `o_proj`      | row-parallel      | input (axis 1)    | `(num_heads/N · head_dim) × H`     | output → **AllReduce(sum)**                  |
| `gate_proj`   | column-parallel   | output (axis 0)   | `H × I/N`                          |                                              |
| `up_proj`     | column-parallel   | output (axis 0)   | `H × I/N`                          |                                              |
| `down_proj`   | row-parallel      | input (axis 1)    | `I/N × H`                          | output → **AllReduce(sum)**                  |

Operations that remain replicated on every rank (cheap relative to MatMul):
LayerNorm/RMSNorm, RoPE, attention softmax, residual add, SiLU, embedding lookup.
SDPA and the KV-cache are **not** explicitly parallelized —
they fall out for free because Q/K/V are already head-sharded and the cache
shape constants get patched to `local_kv_heads`.

The vocabulary projection is sharded too, along the vocabulary axis, and its
slices are collected into rank 0 by a `TPGather` rather than summed:
each rank owns a distinct band of logits, so there is nothing to reduce.
Only rank 0's buffer is written, because that is the only rank whose outputs the infer
request reads. `TP_DISABLE_LM_HEAD_SHARDING` turns this off.

**Collectives per inference**: `2 × num_layers` AllReduce calls — one after
attention `o_proj`, one after MLP `down_proj` — plus one Gather for the
vocabulary projection when it is sharded. The exact count comes from `GraphRewriter::count_collectives`.
It is baked into the compiled blob so an imported model can size its coordinator without re-analyzing the graph.
The AllReduce count is the lower bound for Megatron-style TP and cannot be reduced without changing the parallelism scheme.

## Data Flow Inside a Decoder Layer

```
  X (replicated on rank 0 and rank 1)
   │
   ├──────────────────────────────┬──────────────────────────
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
   └──────  TPAllReduce (sum) on device USM ──────┐
            via TPDeviceCoordinator               │
   ┌──────────────────────────────┬───────────────┘
   ▼                              ▼
 + residual                     + residual
 LayerNorm                      LayerNorm
   │                              │
 gate, up_proj column-shard     gate, up_proj column-shard
 SiLU * up                      SiLU * up
 down_proj row-shard            down_proj row-shard
   │  partial sum                 │  partial sum
   ▼                              ▼
   └──────  TPAllReduce (sum) ────┘
   │                              │
 + residual                     + residual
   ▼                              ▼
 (next layer)                   (next layer)
```

## Component Map

The plugin builds the pieces but keeps none of them: everything is moved into
the `CompiledModel`, which is what owns the topology for as long as the model
is alive.

```
Plugin (IPlugin)                       stateless apart from its own config
  │   creates the shared L0 context, the coordinator and the per-rank
  │   intel_gpu models, then moves all of them into:
  ▼
CompiledModel (ICompiledModel)
  ├── TPL0SharedContext                one ze_context over all N devices.
  │                                    Declared first, destroyed last: every
  │                                    USM allocation and every rank network
  │                                    lives inside it.
  ├── TPDeviceCoordinator              per-rank L0 queues, modules, kernels
  │     ├── Plan[2 per collective]     recordings, event pools, command lists
  │     ├── Rendezvous[per collective] enter barrier + record gate
  │     ├── ScratchArena               one device-USM staging block per rank
  │     └── watchdog thread            started on the first spliced collective
  ├── rank_compiled[N]                 intel_gpu ICompiledModel per rank; each
  │                                    holds a CollectiveCommRegistry injected
  │                                    through set_property()
  ├── CacheController                  paged-attention cache, one per model,
  │                                    shared by every infer request
  ├── sharded_state_ids[]              which variables hold a KV-cache slice
  │                                    rather than a replica
  ├── inference mutex                  one outer inference at a time; rank
  │                                    execution inside it stays parallel
  └── RankWorkers                      one thread per non-zero rank, created on
                                       first use and kept for the life of the
                                       model. Declared last, joined first.
        ▲
        │ borrowed by every request, owned by none of them
        │
InferRequest (ISyncInferRequest)       one per create_infer_request()
  ├── m_rank_requests[N]               per-rank IAsyncInferRequest
  ├── m_fanout_states[]                FanOutVariableState: one user-visible
  │                                    state over N per-rank states, gathering
  │                                    and scattering the sharded ones
  └── m_input_stage[][]                host staging for small user inputs
```

During `infer()`:

```
rank 0 runs on the calling thread, ranks 1..N-1 on RankWorkers
    ↓ a rank's graph reaches a TPAllReduce or TPGather node
intel_gpu OCL impl → that network's CollectiveCommRegistry → coordinator()
    ↓ the model's own command list is passed along
TPDeviceCoordinator::allreduce / gather_to_root(..., model_queue)
```

## Source Tree (current)

```
src/plugins/tp_gpu/
├── CMakeLists.txt
├── docs/                                   ← this directory
├── include/tp_gpu/
│   ├── op/                                 ← TPAllReduce, TPGather op headers,
│   │                                         registered in a private "tp_gpu"
│   ├── options.inl                         ← the option table (X-macro)
│   ├── internal_properties.hpp             ← property keys for those options
│   ├── tp_config.hpp                       ← TPConfig: typed getters, validation
│   ├── tp_debug.hpp                        ← logging macros, NS/Stopwatch/
│   │                                         ScopedTime/DumpPeriod
│   └── tp_device_coordinator.hpp           ← TPDeviceCoordinator public API
├── cmake/embed_kernels.cmake               ← bakes the .cl into the plugin
├── src/
│   ├── op/                                 ← TPAllReduce, TPGather op classes
│   ├── kernels/allreduce_sum.cl            ← reduce kernel source
│   ├── plugin.{hpp,cpp}                    ← IPlugin entry point; builds
│   │                                         shared L0 context + coordinator
│   ├── compiled_model.{hpp,cpp}            ← ICompiledModel, export/import
│   ├── infer_request.{hpp,cpp}             ← ISyncInferRequest, rank fan-out
│   ├── rank_workers.hpp                    ← persistent worker thread per rank
│   ├── cache_controller.{hpp,cpp}          ← KV-cache control inputs per rank
│   ├── graph_rewriter.{hpp,cpp}            ← analyze + per-rank rewrite
│   ├── tp_blob.hpp                         ← blob container format helpers
│   ├── tp_config.cpp                       ← device-name resolution
│   ├── tp_l0_shared_context.hpp            ← RAII for the shared ze_context
│   ├── tp_ze_throw.hpp                     ← ZE_THROW helper
│   ├── tp_device_coordinator.cpp           ← L0 plans, cmdlists, kernels, exec
│   └── version.cpp                         ← plugin version
└── tests/
    ├── common/tp_test_models.hpp           ← synthetic blocks, shared by suites
    ├── unit/                               ← ov_tp_gpu_unit_tests: rewriter,
    │                                         TPDeviceCoordinator
    └── functional/                         ← ov_tp_gpu_func_tests: accuracy,
                                             paged attention, serialization
```

`tp_embedded_kernels.h` is generated into the build tree from `src/kernels/allreduce_sum.cl`.

## TPDeviceCoordinator (Device-side Collectives)

The coordinator owns per-rank L0 resources and executes the cross-device
sum on USM-device memory. Every rank records and submits its own commands;
no rank drives the others.

### Per-rank state

For every rank, the coordinator creates:
- A compute queue on the compute ordinal. There is no separate copy queue:
  the copies and the reduce kernel share one command list per collective.
- An `allreduce_sum_f16` and `allreduce_sum_f32` kernel compiled from
  `src/kernels/allreduce_sum.cl`. The kernel module is built via OpenCL
  (`clBuildProgram`) on the device matched by UUID, then re-loaded into
  the shared L0 context as `ZE_MODULE_FORMAT_NATIVE`.
- The GPU timer period and timestamp mask, read once, used to turn kernel
  timestamps into nanoseconds.

### Plan caching

Each collective site (`collective_id`) owns two `Plan`s that alternate by
the parity of the rendezvous generation. Two, because a rank that has left a
collective may reach the same one again while a neighbour is still executing
the previous instance, and the ring writes into a neighbour's staging: the
recording being laid down and the one still running must not share a command
list, an event or a byte of the arena.

A plan owns its event pools, its per-rank command lists, and the signature the
recording was built from:

| Part of the signature            | Why it is there                                 |
|----------------------------------|-------------------------------------------------|
| `in_ptrs[N]`, `out_ptrs[N]`      | the addresses baked into the recording          |
| `in_ids[N]`, `out_ids[N]`        | the driver's allocation ids for those addresses |
| `n`, `dtype`                     | payload shape                                   |
| `recorded_scratch_generation[N]` | which staging arena the recording used          |

The allocation ids are not redundant. intel_gpu frees and reallocates network
buffers between inferences, and a new allocation landing on an old address used
to make the signature check report a hit; the next submit then walked a
destroyed `GraphicsAllocation`. Comparing the driver's id alongside the address
catches that.

A recording is reused whenever the signature and the arena generation both
still hold, which is the overwhelming majority of calls. It is re-recorded when
either moves.

### Staging arena

One device-USM allocation per rank, shared by every plan and sized to the
largest payload seen so far (`payload_capacity_bytes`). Growth drains every
rank's queue first, because the staging addresses are baked into recordings.
The arena is doubled by `plan_buffers()` for the same reason the plans are.

- TP=2: one payload-sized region per rank per buffer.
- Ring / halving (N>2): `N` chunk slots per rank per buffer, so concurrent
  steps never share a slot. `chunk_capacity_bytes` is the stride.

`payload_capacity_bytes` and `generation` are atomic: ranks read them outside
the scratch lock to decide for themselves whether the arena is about to move.

### Executing a collective

```
every rank
  |
  | enter barrier: publish (in, out, alloc ids), wait for all ranks
  |
  | record gate:
  |   rank 0  settles the signature and grows the arena if needed,
  |           then bumps record_gen
  |   others  skip the gate entirely when they can see that neither the
  |           signature nor the arena moved; otherwise wait for record_gen
  |
  | record own commands if this rank's recording is stale
  |
  | splice: zeCommandListImmediateAppendCommandListsExp into the queue
  |         intel_gpu already runs the model on, then return
  |
  └── no exit barrier: the two alternating plan buffers make one unnecessary
```

The splice is what makes the collective asynchronous: the queue is in-order,
so the recording lands after whatever produced `in_dev` and before whatever
reads `out_dev`, and the host goes on dispatching the rest of the model while
the devices work through it. The only host wait is for the *previous* splice of
the same command list, which the extension forbids re-appending while in
flight; with two buffers alternating, that has normally long finished.

Without the splice extension, or under `TP_FORCE_SYNC_COLLECTIVE`, the
collective runs on the coordinator's own queues and the call blocks until the
queue drains.

Because nothing on the host waits for the device any more, a rank that dies
would hang the group with no deadline to catch it. A watchdog thread, started
on the first spliced collective, polls the completion events and aborts the
group if nothing advances within `COMMUNICATION_TIMEOUT_MS`.

### Schedules

The schedule is chosen per recording from the world size and the payload:

| Schedule          | When                                                    | Steps       | Per-rank traffic |
|-------------------|---------------------------------------------------------|-------------|------------------|
| pair              | `N == 2`                                                | 1 exchange  | one payload      |
| recursive halving | `N` a power of two and payload ≤ `TP_HALVING_MAX_BYTES` | `2·log2(N)` | `1.5·S`          |
| ring              | otherwise                                               | `2·(N-1)`   | `1.5·S`          |

Halving and the ring move the same bytes; what halving buys is fewer steps and
fewer round trips, which is what decode-sized payloads are made of. At prompt
sizes it loses, because both partners of a pair push across the same link at
once while the ring spreads the traffic one way around the loop — hence the
payload ceiling.

Source-side enqueue is mandatory throughout: a peer-pull memcpy silently
transfers zeros on the validated driver.

### Gather

`gather_to_root` is the odd one out. Every rank copies its slice into a
column band of rank 0's buffer with one strided region copy — the slices are
strided in the destination, not contiguous. Ranks write disjoint columns, so
there is nothing to order between them, but the root must not read before they
land: every other rank signals `ev_gather[r]` and the root's own recording
waits on all of them. Unlike allreduce, gather keeps an exit barrier.

## intel_gpu Integration

The TP plugin contributes two OCL primitives to intel_gpu,
`tp_allreduce` and `tp_gather`, implementing the in-graph `TPAllReduce` and `TPGather` ops.
Both:
- Have `is_cpu() == false`, which forces intel_gpu to allocate inputs and
  outputs in `usm_device`. This is required for cross-PCIe peer copies.
- During `execute()`, read the immutable `collective_id` / `rank` carried by
  the primitive, resolve the coordinator from the owning network's
  `CollectiveCommRegistry`, and hand it the model's own command list so the
  recording is spliced there. Because the primitive stores only PODs, it
  serializes cleanly and the registry can be re-injected after `import_model`.

There is no CPU fallback: the collective needs a shared L0 context spanning
every rank's device, and without one the plugin fails at `compile_model`
rather than silently degrading.

## Compiled Blob

`CompiledModel::export_model` writes a container around one intel_gpu
blob per rank:

```
char[8]              "OVTPGPU" string
uint32               version
uint32               world_size
uint32               num_collectives
uint32               num_sharded_states
[world_size]         device name      : uint32 length + bytes
[num_sharded_states] variable id      : uint32 length + bytes
[world_size]         rank blob        : uint64 length + bytes
```

The weight shards are already baked into each rank's blob, so no shard
metadata is stored. `num_collectives` is, because at import time there is
no graph left to analyze and the coordinator has to be sized anyway.
The sharded state ids are there for the same reason: the infer request has to know
which states are per-rank slices rather than replicas.

`Plugin::import_model` validates the header, builds its own shared L0
context over the recorded devices, hands each rank blob to intel_gpu
through that rank's remote context, and re-injects a fresh
`CollectiveCommRegistry` with `set_property` — the same mechanism used
after `compile_model`. The resulting `CompiledModel` has no `ov::Model`
and derives its ports from rank 0.

Everything read back is untrusted input (a stale, truncated or corrupt
cache entry), so each length is bounds-checked before it is used to size
an allocation or move the stream. A blob is bound to its topology:
importing it against a different device set is rejected.

`ov::cache_dir` works because the plugin advertises
`ov::device::capability::EXPORT_IMPORT` and answers
`ov::internal::caching_properties` — the GPU list plus `TP_SIZE` and
`DEVICE_IDS`, so two topologies cannot collide on one hash. Values of GPU
caching properties are aggregated across the rank devices and joined with `;`.

## Lifetime and Threading

- `TPL0SharedContext` and `TPDeviceCoordinator` are owned by the
  `CompiledModel`: the plugin builds them and moves them
  in. The shared context is declared first so it is destroyed last, after the
  coordinator and after every rank network that allocated inside it.
- `TPDeviceCoordinator` is shared via `std::shared_ptr` between the
  `CompiledModel` and the `CollectiveCommRegistry` held by each per-rank
  intel_gpu network, so it outlives any single inference.
- `RankWorkers` belongs to the `CompiledModel`, not to a request: one thread
  per non-zero rank, created on first use and reused by every infer request of
  that model. Starting them per inference costs tens of microseconds of
  start-up skew, and that skew would be paid again at every collective.
  It is declared last so its threads are joined before anything they touch
  goes away.
- `infer_request.cpp` dispatches ranks 1..N-1 to those workers and runs rank 0
  on the calling thread.
- `CompiledModel::lock_inference()` serializes outer inferences: the
  rendezvous slots, the recorded command lists and the scratch arena support
  one at a time. Rank execution inside that inference stays parallel.
- Inside `TPDeviceCoordinator::allreduce`, ranks meet at an enter barrier
  over a `std::condition_variable` with generation counters, then pass a
  record gate that rank 0 releases. There is no exit barrier; `gather_to_root`
  keeps one.

## Options

Every knob is a plugin option declared in one table `include/tp_gpu/options.inl`,
where an X-macro gives each entry a typed getter, a default, validation,
and support for both the environment and a JSON config file.

The table has three tiers:

| Tier             | Reachable through                    | Exists in                            |
|------------------|--------------------------------------|--------------------------------------|
| release          | public API, environment, config file | every build                          |
| release-internal | environment, config file             | every build                          |
| debug            | environment, config file             | only `-DENABLE_TP_GPU_DEBUG_CAPS=ON` |

The debug tier is compiled out rather than ignored: without the flag those
options have no getters, so the code they guard folds away instead of costing a
branch on the per-collective path.

The options that change the architecture described above are named where they
apply — `TP_DISABLE_LM_HEAD_SHARDING` under sharding, `TP_HALVING_MAX_BYTES`
and `TP_ENABLE_HALVING` under schedules, `TP_FORCE_SYNC_COLLECTIVE` under execution.
For the full list see:
Public properties: [USER_GUIDE.md § Configuration Properties](USER_GUIDE.md#configuration-properties)
Dev properties: [DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables)

## Known Limitations

- The rewriter anchors on `PagedAttentionExtension` or
  `ScaledDotProductAttention` and requires the projections to be MatMuls
  over constant-derived weights. Architectures that fuse QKV differently,
  or feed a weight from a live parameter, are not recognized.
- Remote contexts supplied by the user are not supported: the plugin
  always creates its own shared L0 context.
- Weightless caching is not wired up, so a cache entry carries the full weights.
- `ov::num_streams` is not honoured: one stream worker per compiled model.
- A failed collective is terminal. The device queues may still hold
  unfinished work, so the coordinator refuses every later call rather than
  running on top of unknown state; the compiled model has to be rebuilt.

## Pointers

- Detailed correctness/perf debugging steps and tunables: see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
- External API and configuration: see [USER_GUIDE.md](USER_GUIDE.md).

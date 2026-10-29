# Tensor Parallel Plugin — Architecture

## Overview

The Tensor Parallel (`TP`) plugin is an OpenVINO meta-plugin that
runs a single transformer model across multiple Intel discrete GPUs using
**Megatron-style tensor parallelism**: weights of large MatMuls are sharded
across ranks, partial sums are summed back via in-graph AllReduce
collectives, and the cross-device synchronization is implemented directly on
device memory through a shared Level Zero context. The collectives run in
the intel_gpu plugin; this plugin talks to them only through the private
collective contract (`openvino/runtime/tp/collective_contract.hpp`).

```
┌──────────────────────────────────────────────────────────────┐
│                     User Application                         │
│   core.compile_model(model, "TP:GPU.0,GPU.1", config)        │
│   request = compiled.create_infer_request()                  │
│   request.infer()                                            │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                         TP Plugin                            │
│                                                              │
│  compile_model():                                            │
│    1. Parse config (device list from "TP:<devices>").        │
│    2. Create one GPU remote context per rank with the TP     │
│       parameters of the collective contract: device group +  │
│       the rank's endpoint of an InProcessExchange. Each      │
│       context publishes its setup payload and returns.       │
│    3. Analyze the model: walk back from every attention op   │
│       to find the Q/K/V/O projections and the MLP            │
│       gate/up/down — produces a ShardingPlan.                │
│    4. For each rank: clone the model, slice weight Constants │
│       column- or row-parallel, patch num_kv_heads constants  │
│       and reshape shapes, insert TPAllReduce after each      │
│       row-parallel MatMul and a TPGather after the sharded   │
│       vocabulary projection, validate the cloned graph.      │
│    5. Compile each per-rank model with its rank context. The │
│       GPU plugin binds the model to the context's            │
│       communicator; the first binding collects the payloads  │
│       and connects the group.                                │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│              CompiledModel (ICompiledModel)                  │
│                                                              │
│  Owns everything the ranks share:                            │
│    - rank_compiled[N]     per-rank intel_gpu model; each     │
│                           holds its rank context, which owns │
│                           the rank's communicator            │
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
│       intel_gpu calls the network's TP channel, which hands  │
│       the recorded collective to the model's own command     │
│       queue.                                                 │
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
vocabulary projection when it is sharded. The exact count comes from `GraphRewriter::count_collectives`
and is recorded in the compiled blob; the GPU plugin counts the collectives of each rank model itself when it
binds the model.
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
            via the GPU TP communicator           │
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
is alive. The collective state lives in the GPU plugin, reached through the
rank contexts.

```
Plugin (IPlugin)                       stateless apart from its own config
  │   creates the rank contexts (create_context with ov::tp::internal
  │   parameters) and the per-rank intel_gpu models, then moves the
  │   models into:
  ▼
CompiledModel (ICompiledModel)
  ├── rank_compiled[N]                 intel_gpu ICompiledModel per rank; each
  │                                    holds its rank context
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

intel_gpu, per rank (src/plugins/intel_gpu/src/runtime/ze/tp/)
RemoteContextImpl (TP context)
  ├── RankCommunicator                 this rank's end; publishes its payload at
  │     │                              creation, connects at the first bind
  │     ├── TPL0SharedContext          one ze_context over all N devices, cached
  │     │                              per device list; the rank's device and
  │     │                              engine sit on it
  │     └── GroupState                 the group's collective resources, one per
  │           │                        group in the process, shared by every rank
  │           ├── Mailbox                    call rendezvous, slots[2][N]
  │           ├── EventSet[2], GatherEvents[2] peer events, one set per call parity
  │           ├── Staging[N]                 device-USM staging, owned per rank
  │           ├── queue per rank             the synchronous fallback
  │           └── watchdog thread            started on the first splice; polls the
  │                                          splices the recordings register
  └── network (per compiled model)     holds a Channel: group state + rank
        └── tp_allreduce / tp_gather impl, per primitive instance
              └── recorder                 command lists[2], completion events[2],
                                         signature[2], the schedules
```

During `infer()`:

```
rank 0 runs on the calling thread, ranks 1..N-1 on RankWorkers
    ↓ a rank's graph reaches a TPAllReduce or TPGather node
intel_gpu impl → its recorder, over that network's TP Channel
    ↓ GroupState::begin_allreduce / begin_gather (call rendezvous)
    ↓ record if stale, splice into the model's own command list
```

## Source Tree (current)

```
src/plugins/tp/
├── CMakeLists.txt
├── docs/                                   ← this directory
├── include/tp/
│   ├── options.inl                         ← the option table (X-macro)
│   ├── internal_properties.hpp             ← property keys for those options
│   ├── tp_config.hpp                       ← TPConfig: typed getters, validation
│   └── tp_debug.hpp                        ← logging macros, NS/Stopwatch/
│                                             ScopedTime/DumpPeriod
├── src/
│   ├── plugin.{hpp,cpp}                    ← IPlugin entry point; creates the
│   │                                         rank contexts
│   ├── compiled_model.{hpp,cpp}            ← ICompiledModel, export/import
│   ├── infer_request.{hpp,cpp}             ← ISyncInferRequest, rank fan-out
│   ├── rank_workers.hpp                    ← persistent worker thread per rank
│   ├── cache_controller.{hpp,cpp}          ← KV-cache control inputs per rank
│   ├── graph_rewriter.{hpp,cpp}            ← analyze + per-rank rewrite
│   ├── tp_blob.hpp                         ← blob container format helpers
│   ├── tp_config.cpp                       ← device-name resolution
│   └── version.cpp                         ← plugin version
└── tests/
    ├── common/tp_test_models.hpp           ← synthetic blocks, shared by suites
    ├── unit/                               ← ov_tp_unit_tests: rewriter
    └── functional/                         ← ov_tp_func_tests: accuracy,
                                              paged attention, serialization,
                                              TP contexts

src/plugins/intel_gpu/   (compiled only with ENABLE_TP)
├── src/runtime/ze/tp/
│   ├── rank_communicator.{hpp,cpp}         ← RankCommunicator, Channel, the
│   │                                         process registry of groups
│   ├── group_state.{hpp,cpp}               ← GroupState: rendezvous, staging,
│   │                                         events, watchdog, profiling report
│   ├── tp_l0_shared_context.{hpp,cpp}      ← shared ze_context + its cache
│   ├── collective_options.hpp              ← CollectiveOptions (GPU_TP_*)
│   └── tp_debug.hpp                        ← timers and report helpers
├── src/graph/impls/ocl_v2/tp/
│   ├── tp_allreduce.{hpp,cpp}              ← AllReduce impl
│   ├── tp_gather.{hpp,cpp}                 ← Gather impl (copies only)
│   ├── tp_allreduce_recorder.{hpp,cpp}     ← pair / ring / halving schedules
│   ├── tp_gather_recorder.{hpp,cpp}        ← slices through the root's staging
│   └── tp_recording.{hpp,cpp}              ← lists, completion events, splice
├── src/graph/impls/ocl_v2/tp_allreduce.cl  ← the reduction kernel (codegen
│                                          reads .cl from the ocl_v2 root only)
├── src/plugin/ops/tp_{allreduce,gather}.cpp       ← op → primitive
└── tests/unit/module_tests/ze/tp_collectives_test.cpp
```

The collective ops themselves (`TPAllReduce`, `TPGather`) live in
`src/common/transformations/include/ov_ops/tp_collective.hpp`.

## Device-side Collectives

The collectives run on USM-device memory across the devices of one shared L0
context. Every rank records and submits its own commands; no rank drives the
others. The work is split in two:

- `GroupState` holds the group's resources: the call rendezvous, every rank's
  staging, the events the ranks signal each other, a queue per rank for the
  synchronous fallback, failure handling and the watchdog. It knows nothing of
  schedules. The first rank model bound to a TP context creates it, the other
  ranks find it through the process registry.
- The recorders of the `tp_allreduce` and `tp_gather` impls own everything
  that runs over those resources: the schedules, the command lists, their
  completion events and the signatures they were recorded from, one set per
  primitive instance. A recorder registers its completion events with the
  group, which polls them (watchdog), signals them (abort) and waits for the
  rank's last one before moving its staging.

The group makes every collective's lists and completion events when it is
created, and a recorder takes its set over on first use. Creating them costs
about a millisecond per collective and rank (measured: 256 sets on four ranks
took 255 ms), which would otherwise land on the first inference.

### Per-rank state

For every rank, the group creates:
- A compute queue on the compute ordinal, used only by the synchronous
  fallback. There is no separate copy queue: the copies and the reduce kernel
  share one command list per collective.
- The GPU timer period and timestamp mask, read once, used to turn kernel
  timestamps into nanoseconds.

Nothing here builds kernels. The reduction kernel (`tp_allreduce.cl`,
f16/bf16/f32) belongs to the compiled model: the `ocl::tp_allreduce` impl
builds it through the GPU kernel pipeline with the rest of the model, so it is
stored in the cache blob and `import_model()` does not rebuild it. The kernel
widens both operands to f32, adds them and rounds once to the element type, so
every reduction step is one f32 addition; the recorder launches it in
`append_reduce()` (argument order and 8 elements per work item).

### Recordings

Every rank counts its allreduce calls and its gather calls separately; the
parity of that count picks one of a recorder's two command lists, one of two
staging halves and one of two event sets. Two, because a rank that has left a
collective may reach the next one while a neighbour is still executing the
previous instance, and peers write into each other's staging: the recording
being laid down and the one still running must not share a command list, an
event or a byte of staging. Gathers keep their own count because a gather
waits on nobody: counted together with the allreduces, an allreduce on one side
of a gather could share a parity with the allreduce on the other side while the
first is still running.

Per parity a recorder keeps the signature its list was recorded from. A rank's
commands depend only on its own buffers and kernel, the call's shape, and where
the peers it writes to have their staging, so each rank checks and re-records
its own recording alone; nothing waits for rank 0.

| Part of the signature             | Why it is there                                 |
|-----------------------------------|-------------------------------------------------|
| `in`, `out`                       | this rank's addresses baked into the recording  |
| `in_id`, `out_id`                 | the driver's allocation ids for those addresses |
| `kernel`                          | this rank's reduction kernel                    |
| `n`, `dtype`                      | payload shape                                   |
| `staging[N]`, `staging_ids[N]`    | every rank's staging half for this parity, as published in the mailbox |
| gather: `rows`, `slice_elems`, `root_staging`, `root_staging_id` | the slice shape and where the root takes the slices |

The allocation ids are not redundant. intel_gpu frees and reallocates network
buffers between inferences, and a new allocation landing on an old address used
to make the signature check report a hit; the next submit then walked a
destroyed `GraphicsAllocation`. Comparing the driver's id alongside the address
catches that.

A recording is reused whenever its signature still holds, which is the
overwhelming majority of calls.

### Staging

One device-USM allocation per rank, owned by that rank, shared by every
collective, sized for the largest payload the rank has seen and split into two
equal halves, one per call parity. The recorder says how large a half has to
be for its schedule; the group only allocates. A rank publishes the address and
allocation id of its half
with every call, so it grows on its own: before it frees the old allocation it
waits for its own last splice, which itself waited for every peer write into
that staging, and peers cannot write the new one before they have read its
address in the rendezvous. Recordings that still point at the old staging see
the id change in their signature.

- TP=2: one payload-sized region per half, aligned to 256 bytes.
- Ring / halving (N>2): `N` chunk slots per half, so concurrent steps never
  share a slot. Halving lays one region per scatter step and one per gather
  step over the same space, sized from the collective's own payload.

No schedule writes into a peer's model buffers: peers write staging only, and
each rank delivers into its own output. Remote writes into a buffer the
receiving rank's own kernels had just written were measured to lose against
them now and then (the halving gather used to do this: about one decode run in
twelve on four ranks produced a different token).

### Executing a collective

```
every rank
  |
  | grow own staging if the payload does not fit (allreduce)
  |
  | call rendezvous: publish {seq, kind, collective_id, n, dtype, rows,
  |   own staging half (allreduce) or the root's gather staging half} into
  |   mailbox slots[seq & 1][rank]; the last arrival releases the others.
  |   A mismatch in seq, kind, id or shape aborts the group and throws.
  |
  | record own commands if this rank's signature moved
  |
  | splice: zeCommandListImmediateAppendCommandListsExp into the queue
  |         intel_gpu already runs the model on, then return
  |
  └── no exit barrier: the alternating slots, command lists, staging halves
      and event sets make one unnecessary
```

A mailbox slot set is rewritten only at call `k + 2`, which a rank reaches
after the rendezvous of `k + 1`, i.e. after every peer has published `k + 1`
and therefore read the slots of `k`.

The splice is what makes the collective asynchronous: the queue is in-order,
so the recording lands after whatever produced `in_dev` and before whatever
reads `out_dev`, and the host goes on dispatching the rest of the model while
the devices work through it. The only host wait is for the *previous* splice of
the same command list, which the extension forbids re-appending while in
flight; with two buffers alternating, that has normally long finished.

Without the splice extension, or under `GPU_TP_FORCE_SYNC_COLLECTIVE`, the
collective runs on the group's own queue of the rank and the call blocks until
the queue drains.

The impl returns no event of its own: the stream is in-order, so whatever
reads the output is queued behind the splice. Only a network output gets a
marker (`enqueue_marker`) that completes with the collective.

Because nothing on the host waits for the device any more, a rank that dies
would hang the group with no deadline to catch it. A watchdog thread, started
on the first spliced collective, polls the completion events the recordings
registered and aborts the group if nothing advances within
`COMMUNICATION_TIMEOUT_MS`. It then signals every event of the group from the
host -- peer events, gather acks and completion events -- so the waiting queues
drain. What they produced is garbage, so the inference that ran into it throws
once its stream has drained (`Graph::throw_if_collectives_failed()` in
`SyncInferRequest::wait()`), every later collective throws at once, and binding
another model to the group fails.

### Schedules

The schedule is chosen per recording from the world size and the payload:

| Schedule          | When                                                    | Steps       | Per-rank traffic |
|-------------------|---------------------------------------------------------|-------------|------------------|
| pair              | `N == 2`                                                | 1 exchange  | one payload      |
| recursive halving | `N` a power of two and payload ≤ `GPU_TP_HALVING_MAX_BYTES` | `2·log2(N)` | `1.5·S`     |
| ring              | otherwise                                               | `2·(N-1)`   | `1.5·S`          |

Halving and the ring move the same bytes; what halving buys is fewer steps and
fewer round trips, which is what decode-sized payloads are made of. At prompt
sizes it loses, because both partners of a pair push across the same link at
once while the ring spreads the traffic one way around the loop — hence the
payload ceiling.

Source-side enqueue is mandatory throughout: a peer-pull memcpy silently
transfers zeros on the validated driver.

### Gather

The gather collects every rank's `rows x slice` matrix as a column band of
rank 0's output, and works like the allreduce: no rank writes into another
rank's model buffer. The root keeps a gather staging of its own (two halves by
gather parity, one slab per peer) and publishes its half in the rendezvous.
Every other rank copies its slice there contiguously and signals `ev[r]`; the
root copies its own band locally, waits for all `ev[r]`, then copies each slab
into its band with one strided region copy.

Unlike the allreduce, the peers depend on nothing the root produces, so the
parity argument does not stop a peer from refilling a half the root has not
copied out yet -- two gathers in a row would do it. The root therefore signals
`ack[r]` once the slabs are out, and rank `r` waits for it before writing the
same half again; the acks are created signalled, so the first gather of each
parity does not wait. A test holding the root's device back corrupts every
later instance without the ack.

## intel_gpu Integration

The collectives live in intel_gpu, compiled only with `ENABLE_TP`: two OCL
primitives, `tp_allreduce` and `tp_gather`, implementing the in-graph
`TPAllReduce` and `TPGather` ops, and the communicator under
`src/runtime/ze/tp/`. The TP plugin reaches them through the private
collective contract only.

### TP context

`create_context()` with `ov::tp::internal::device_group` and
`ov::tp::internal::groups` (group 0, this rank's `IExchange` endpoint) builds
the context of one rank:
- the device-group L0 context over every device of the group, created once
  per device list and shared by every rank's context in the process;
- a remote context whose device and engine sit on that shared context;
- this rank's `RankCommunicator`, which publishes its setup payload
  (header, process id, a process-unique communicator id) and returns without
  waiting.

Other accepted parameters: `COMMUNICATION_TIMEOUT_MS`, `DEVICE_ID` and the
`GPU_TP_*` options. `ov_internal_tp_channels` other than 1 and
`ov_internal_tp_comm_buffer_size` are refused until they are implemented;
any other parameter is an error.

### Binding

`compile_model()` and `import_model()` with a TP context bind every stream
graph of the model to the context's communicator (`Graph::build`):
- every collective's rank and world size must match the context, and the ids
  must number the model's collectives once each;
- the first binding collects the payloads, checks them, and finds or creates
  the group state (`GroupState`) in a process registry keyed by the rank-0
  communicator id; every rank model must have the same number of collectives;
- one model per channel at a time, and one channel per context, so a second
  stream fails the compilation;
- a model with collectives compiled for an ordinary context fails the
  compilation;
- a TP context forces the in-order queue; an explicit out-of-order request
  fails.

The network then holds a `Channel` (group state + rank).

### Primitives

Both primitives:
- Have `is_cpu() == false`, which forces intel_gpu to allocate inputs and
  outputs in `usm_device`. This is required for cross-PCIe peer copies.
- During `execute()`, read the `collective_id` carried by the primitive
  (`collective_id` is the op's position among the TP collectives of the
  program, assigned by the GPU `ProgramBuilder`; transitional until calls
  carry a sequence number) and run their recorder over the network's
  `Channel` with the model's own command list, so the recording is spliced
  there. The recorder's lists and events belong to the primitive instance; a
  cloned impl starts without them. Because the primitive stores only PODs, it
  serializes cleanly, and an imported model is bound the same way as a
  compiled one.

There is no CPU fallback: the collective needs a shared L0 context spanning
every rank's device, and without one the compilation fails rather than
silently degrading.

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
metadata is stored. `num_collectives` is informational: the GPU plugin counts
the collectives of each imported rank model when it binds it.
The sharded state ids are there because the infer request has to know
which states are per-rank slices rather than replicas.

`Plugin::import_model` validates the header, creates the rank contexts over
the recorded devices and hands each rank blob to intel_gpu with that rank's
context, which binds it exactly as after `compile_model`. The resulting
`CompiledModel` has no `ov::Model` and derives its ports from rank 0.

Everything read back is untrusted input (a stale, truncated or corrupt
cache entry), so each length is bounds-checked before it is used to size
an allocation or move the stream. A blob is bound to its topology:
importing it against a different device set is rejected.

`ov::cache_dir` works because the plugin advertises
`ov::device::capability::EXPORT_IMPORT` and answers
`ov::internal::caching_properties` — the GPU list plus the device list
(`ov::device::priorities`), so two topologies cannot collide on one hash. Values of GPU
caching properties are aggregated across the rank devices and joined with `;`.

## Lifetime and Threading

- Each rank's intel_gpu compiled model holds its TP context, the context owns
  the rank's `RankCommunicator`, and the communicators share the group's
  `GroupState` and the `TPL0SharedContext` through `std::shared_ptr`. Every
  recorder holds the group too, so the staging and events its lists name
  outlive it; a recorder waits for its last splice before destroying its lists.
  The shared context outlives the group, and every rank's device and
  engine, because each of them holds it. The TP `CompiledModel` holds no Level
  Zero state of its own.
- `RankWorkers` belongs to the `CompiledModel`, not to a request: one thread
  per non-zero rank, created on first use and reused by every infer request of
  that model. Starting them per inference costs tens of microseconds of
  start-up skew, and that skew would be paid again at every collective.
  It is declared last so its threads are joined before anything they touch
  goes away.
- `infer_request.cpp` dispatches ranks 1..N-1 to those workers and runs rank 0
  on the calling thread.
- `CompiledModel::lock_inference()` serializes outer inferences: the
  rendezvous slots, the recorded command lists and the staging support
  one at a time. Rank execution inside that inference stays parallel.
- Inside `GroupState::begin_allreduce` and `begin_gather`, ranks meet at
  a call rendezvous (a mailbox over a `std::condition_variable`, slots
  alternating by call parity). There is no record gate and no exit barrier.

## Options

Every knob is a plugin option declared in one table `include/tp/options.inl`,
where an X-macro gives each entry a typed getter, a default, validation,
and support for both the environment and a JSON config file.

The table has three tiers:

| Tier             | Reachable through                    | Exists in                            |
|------------------|--------------------------------------|--------------------------------------|
| release          | public API, environment, config file | every build                          |
| release-internal | environment, config file             | every build                          |
| debug            | environment, config file             | only `-DENABLE_TP_DEBUG_CAPS=ON` |

The debug tier is compiled out rather than ignored: without the flag those
options have no getters, so the code they guard folds away instead of costing a
branch on the per-collective path.

The options that change the architecture described above are named where they
apply — `TP_DISABLE_LM_HEAD_SHARDING` under sharding.

The collectives are tuned in the GPU plugin, which runs them: `GPU_TP_*`
options in a separate block of intel_gpu's `options.inl`, compiled only with
`ENABLE_TP` (debug ones also need the GPU debug caps). They are read when a TP
context is created, from the device configuration, the context parameters and
the environment (`OV_GPU_TP_*`). `GPU_TP_HALVING_MAX_BYTES` and
`GPU_TP_ENABLE_HALVING` are named under schedules, `GPU_TP_FORCE_SYNC_COLLECTIVE`
under execution. `TP_VERBOSE`, `TP_PROFILING` and `TP_DUMP_PERIOD` cover this
plugin only; the collectives have `GPU_TP_VERBOSE`, `GPU_TP_PROFILING` and
`GPU_TP_DUMP_PERIOD`. The TP plugin passes no option of a device plugin down:
it does not know which device plugin runs its ranks.

For the full list see:
Public properties: [USER_GUIDE.md § Configuration Properties](USER_GUIDE.md#configuration-properties)
Dev properties: [DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables)

## Known Limitations

- The rewriter anchors on `PagedAttentionExtension` or
  `ScaledDotProductAttention` and requires the projections to be MatMuls
  over constant-derived weights. Architectures that fuse QKV differently,
  or feed a weight from a live parameter, are not recognized.
- Remote contexts supplied by the user are not supported: the plugin
  always creates the rank contexts itself.
- Weightless caching is not wired up, so a cache entry carries the full weights.
- `ov::num_streams` is not honoured: a TP context has one channel, so a rank
  model with more than one stream fails to compile.
- All ranks of a group run in one process.
- A failed collective is terminal. The device queues may still hold
  unfinished work, so the group refuses every later call rather than
  running on top of unknown state; the compiled model has to be rebuilt.

## Pointers

- Detailed correctness/perf debugging steps and tunables: see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
- External API and configuration: see [USER_GUIDE.md](USER_GUIDE.md).

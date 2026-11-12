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
│    1. Parse config (device list from "TP:<devices>"); reject │
│       ov::num_streams other than 1; ask the device plugin    │
│       (collective_support) whether it can serve the group;   │
│       reject models with linear-attention state.             │
│    2. Analyze a clone of the model once (TPLayerAnalyzer):   │
│       block passes annotate the projections, shape constants │
│       and KV-cache variables of every attention and MLP      │
│       block; TPShardingVerifier proves the annotations or    │
│       fails the compilation naming the node. Each rank's     │
│       weights, read off the annotations, are compared with   │
│       its device's free memory (TP_DEVICE_MEMORY)            │
│       and the compilation stops if they do not fit, naming   │
│       the fewest devices that would do.                      │
│    3. Create one remote context per rank with the TP         │
│       parameters of the collective contract: device group +  │
│       the rank's endpoint of an InProcessExchange. Each      │
│       context publishes its setup payload and returns.       │
│    4. For each rank: clone the analyzed model,               │
│       TPWeightShardingPass cuts the annotated constants and  │
│       gives shape constants the rank's sizes,                │
│       TPInsertCollectives inserts TPAllReduce after each     │
│       row-parallel MatMul and a TPGather after the split     │
│       vocabulary projection; annotations are stripped.       │
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
│    - RankWorkers          a thread per rank                  │
│    - CacheController      paged-attention cache              │
│    - device locks         one TP request at a time on these  │
│                           devices, process-wide              │
│    - TP RemoteContext     get_context(): host tensors only   │
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
│    2. Hand every rank to its persistent worker thread; the   │
│       calling thread waits for all of them.                  │
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

For a standard decoder block with hidden size `H`,
`num_heads` query heads, `num_kv_heads` (GQA), and intermediate size `I`,
the analysis shards the following MatMul weights:

| MatMul        | Parallel kind     | Weight slice axis | Shard size                         | Notes                                        |
|---------------|-------------------|-------------------|------------------------------------|----------------------------------------------|
| `q_proj`      | column-parallel   | output (axis 0)   | `H × (num_heads/N · head_dim)`     | whole query groups per rank                  |
| `k_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  | the KV heads of the rank's query heads       |
| `v_proj`      | column-parallel   | output (axis 0)   | `H × (num_kv_heads/N · head_dim)`  |                                              |
| `o_proj`      | row-parallel      | input (axis 1)    | `(num_heads/N · head_dim) × H`     | output → **AllReduce(sum)**                  |
| `gate_proj`   | column-parallel   | output (axis 0)   | `H × I/N`                          |                                              |
| `up_proj`     | column-parallel   | output (axis 0)   | `H × I/N`                          |                                              |
| `down_proj`   | row-parallel      | input (axis 1)    | `I/N × H`                          | output → **AllReduce(sum)**                  |

Splits are in whole units: query groups (or, with fewer KV heads than ranks, equal shares of one
group), KV heads, and for the MLP the coarsest step any of its weights needs (quantization
groups). Every rank takes the same number of units; a world size `N` that does not divide them
fails the compilation, naming the dimension. An uneven split would give the ranks rank models of
different shapes, so a device rewrite that depends on a shape could order the independent
collectives of parallel branches differently on different ranks; the busiest rank would also
pace the group.

**Fewer KV heads than ranks** (`N % num_kv_heads == 0` and `num_heads % N == 0`): every rank
keeps the one KV head its query heads use, `r / (N / num_kv_heads)`, and an equal share of that
head's query group. The `N / num_kv_heads` ranks sharing a KV head compute its K/V projections
alike and keep it in their own KV caches; the GQA expansion of a rank goes to its share of the
group only. Any other configuration with fewer KV heads than ranks is rejected.

**Fused projections** (one `qkv_proj` split into Q, K and V; one `gate_up_proj` split into gate
and up): the weight is cut per part, so every rank holds its share of every part,
and the `VariadicSplit` lengths or `Slice` bounds after it are the rank's; equal chunks (`Split`)
stay equal.

Operations that remain replicated on every rank (cheap relative to MatMul):
LayerNorm/RMSNorm, RoPE, attention softmax, residual add, SiLU, embedding lookup.
SDPA and the KV-cache are **not** explicitly parallelized —
they fall out for free because Q/K/V are already head-sharded and the cache
shape constants get patched to `local_kv_heads`.

The vocabulary projection is sharded too, along the vocabulary axis, and its
slices are collected into rank 0 by a `TPGather` rather than summed:
each rank owns a distinct band of logits, so there is nothing to reduce.
Only rank 0's buffer is written, because that is the only rank whose outputs the infer
request reads. `TP_DISABLE_LM_HEAD_SHARDING` (a release option, read from the environment, part of
the cache key) turns this off. When the LM head shares its weight
with the embedding (tied weights), the embedding table stays whole and the LM head gets a copy of
the chain to cut: the split costs `table / N` more device memory per rank, and in exchange each
rank computes only its share of the vocabulary projection, one of the largest MatMuls of a decode
step.

**Collectives per inference**: `2 × num_layers` AllReduce calls — one after
attention `o_proj`, one after MLP `down_proj` — plus one Gather for the
vocabulary projection when it is sharded. The device plugin counts the collectives of each rank
model itself when it binds the model.
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

## Transformations (L1)

The graph work is a static library, `src/transformations/` (`openvino_tp_transformations`),
that depends on OpenVINO core and the collective ops only. It runs in three stages:

1. **`TPLayerAnalyzer`**, once per model, on a clone of it. Block passes, one per block type,
   recognize a block structurally and annotate it in `rt_info`:
   `TPPagedAttentionBlock`, `TPStatefulSDPABlock` (both also for a fused `qkv_proj`),
   `TPGatedMLPBlock`, `TPFusedGateUpMLPBlock`, `TPDenseMLPBlock`, `TPVocabParallelBlock`.
   A projection gets a `tp_role` (column- or row-parallel, the collective after it); every
   constant to cut gets a `tp_shard` (axis and segments; a segment is split in whole units, can
   be shared by consecutive ranks or split into shares of a group); shape constants get the
   elements that hold a split size, a fused part's length or a fused part's bound; KV-cache
   variables and PagedAttention head counts get the KV-head split. A weight shared with another
   reader (a tied embedding) is given a source of its own first, so cutting it leaves the other
   reader whole. A linear layer no pass claims stays whole on every rank, with a warning naming it
   and its size. Weights are split only when every constant of their chain is of 8 bits and
   more or a 4-bit type; 1, 2, 3 or 6-bit constants fail the compilation, naming the weight.
   Packed 4-bit constants (weights, zero points) are cut at any element, copying
   element by element where a rank's range starts or ends inside a byte. Every weight is cut at
   compile time: one whose decompression chain cannot be cut at the rank boundaries (a boundary
   inside a quantization group, an op in the chain the walk does not know) fails the compilation,
   naming the weight and the reason; an LM head in that case stays whole instead. A boundary
   falls inside a group only for the O projection, whose cut follows the query heads, when a
   rank's query heads x `head_dim` is not a whole number of groups (e.g. a `head_dim` of 64, an
   odd number of query heads per rank and groups of 128). The MLP cuts in whole groups.
2. **`TPShardingVerifier`**, the last pass of the analysis, propagates a state through every
   tensor: Replicated, Split (per-rank ranges along one or more axes), Partial (the sum over
   ranks is the value) or rank-dependent sizes (a shape vector from `ShapeOf` of a split tensor
   or an annotated shape constant, tracked per element). Each op type has a rule; a row-parallel
   role must produce a Partial sum, which becomes Replicated at its `TPAllReduce`; a model output
   must be Replicated. A Split, VariadicSplit or Slice along a split axis is proven by simulating
   every rank with its own lengths or bounds. Anything it cannot prove fails the compilation with
   the node, its type, the input shapes and states, and where each state came from. A split
   with fewer units (whole heads, quantization groups) than ranks fails too, naming the largest
   number of devices that works.
3. **Per rank**, on a clone of the analyzed model: `TPWeightShardingPass(rank, N)` cuts the
   annotated constants (a zero-copy view of a single contiguous range, a copy otherwise) and gives
   the annotated shape elements, variable shapes and `rt_info` head counts the rank's values;
   `TPInsertCollectives(rank, N)` inserts the collectives the roles
   name. The annotations are then stripped.

The analysis also reports what the plugin logs and keeps: the block counts, the geometry, the
KV-cache variables split by head and how many ranks share each KV head.

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
  ├── CacheController                  paged-attention cache, one per model
  │                                    (ov::plugin_managed_cache); binds the
  │                                    cache to every registered request
  ├── sharded_state_ids[]              which variables hold a KV-cache slice
  │                                    rather than a replica
  ├── device locks                     process-wide, one per device (by UUID),
  │                                    taken in one order for a TP request;
  │                                    rank execution inside it stays parallel
  ├── RemoteContext                    get_context(): host tensors through rank
  │                                    0's context; device tensors rejected
  └── RankWorkers                      one thread per rank, created on
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
  └── m_input_stage[][]                small user inputs copied into host
                                       tensors of each rank's own context
                                       (create_host_tensor), read directly

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
every rank runs on its RankWorker; the calling thread dispatches and waits
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
│   ├── properties.hpp                      ← public-facing keys: TP_STRATEGY, COLLECTIVE_BACKEND
│   ├── internal_properties.hpp             ← keys of the internal, tuning and debug options
│   ├── tp_config.hpp                       ← TPConfig: typed getters, validation
│   ├── tp_debug.hpp                        ← logging macros (TP_LOG_*, TP_WARN_ALWAYS, TP_REPORT)
│   └── itt.hpp                             ← ITT domains ov::tp::plugin / ov::tp::inference
├── src/
│   ├── plugin.{hpp,cpp}                    ← IPlugin entry point; creates the
│   │                                         rank contexts
│   ├── compiled_model.{hpp,cpp}            ← ICompiledModel, export/import
│   ├── infer_request.{hpp,cpp}             ← ISyncInferRequest, rank fan-out
│   ├── rank_workers.hpp                    ← persistent worker thread per rank
│   ├── device_locks.hpp                    ← process-wide per-device locks
│   ├── remote_context.{hpp,cpp}            ← the TP context: host tensors only
│   ├── cache_controller.{hpp,cpp}          ← IPluginManagedCache: the PA cache on every rank
│   ├── transformations/                    ← L1, static library openvino_tp_transformations
│   │   ├── include/tp_transformations/     ← annotations, analyzer, verifier, rank passes
│   │   └── src/                            ← block passes (attention, MLP, vocabulary),
│   │                                         block utilities, verifier, rank passes
│   ├── tp_blob.hpp                         ← blob container format helpers
│   ├── tp_config.cpp                       ← device-name resolution
│   └── version.cpp                         ← plugin version
└── tests/
    ├── common/tp_test_models.hpp           ← synthetic blocks, shared by suites
    ├── unit/                               ← ov_tp_unit_tests: annotations, analysis,
    │                                         verifier, rank models (no device)
    └── functional/                         ← ov_tp_func_tests: accuracy, paged attention,
                                              serialization, device names, options; the
                                              rank device type from OV_TP_TEST_DEVICE

src/tests/functional/plugin/shared/{include,src}/behavior/tp/
└── collective_contract.*                   ← L3 conformance tests, instantiated by a device plugin

src/plugins/intel_gpu/   (compiled only with ENABLE_TP)
├── src/runtime/ze/tp/
│   ├── rank_communicator.{hpp,cpp}         ← RankCommunicator, Channel, the
│   │                                         process registry of groups
│   ├── group_state.{hpp,cpp}               ← GroupState: rendezvous, staging,
│   │                                         events, watchdog, profiling report
│   ├── collective_backend.{hpp,cpp}        ← backend names, support checks, AUTO
│   ├── tp_l0_shared_context.{hpp,cpp}      ← shared ze_context + its cache
│   ├── collective_options.hpp              ← CollectiveOptions (GPU_TP_*)
│   └── tp_debug.hpp                        ← timers and report helpers
├── src/graph/impls/ocl_v2/tp/
│   ├── tp_allreduce.{hpp,cpp}              ← AllReduce impl
│   ├── tp_gather.{hpp,cpp}                 ← Gather impl (copies only)
│   ├── tp_allreduce_recorder.{hpp,cpp}     ← pair / ring / halving schedules
│   ├── tp_gather_recorder.{hpp,cpp}        ← bands into the root's output
│   └── tp_recording.{hpp,cpp}              ← lists, completion events, splice
├── src/graph/impls/ocl_v2/tp_allreduce.cl  ← the reduction kernel (codegen
│                                          reads .cl from the ocl_v2 root only)
├── src/plugin/ops/tp_{allreduce,gather}.cpp       ← op → primitive
├── tests/unit/module_tests/ze/tp_collectives_test.cpp
└── tests/functional/tp/, shared_tests_instances/behavior/tp/  ← GPU TP tests, the L3 suite for GPU
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
created, and a recorder takes its set over on first use. Creating them takes time for every
collective and rank, which would otherwise land on the first inference.

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
does not order the peers against each other (each waits only for the root):
counted together with the allreduces, an allreduce on one side
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
| `chunk_elems`, `halving`          | how the payload is split and which schedule each chunk takes |
| `staging[2][N]`, `staging_ids[N]` | every rank's two staging halves, as published in the mailbox |
| gather: `rows`, `slice_elems`, `root_out`, `root_out_id` | the slice shape and the root's output the bands go to |

The allocation ids are not redundant. intel_gpu frees and reallocates network
buffers between inferences, and a new allocation landing on an old address used
to make the signature check report a hit; the next submit then walked a
destroyed `GraphicsAllocation`. Comparing the driver's id alongside the address
catches that.

A recording is reused whenever its signature still holds, which is the
overwhelming majority of calls.

### Staging

One device-USM allocation per rank, owned by that rank, shared by every
collective and split into two equal halves, one per call parity. The recorder
says how large a half has to be for its schedule; the group only allocates. A
rank publishes the address, half size and allocation id of its staging with
every call.

By default the staging is sized for the largest payload the rank has seen, so
it grows on its own: before it frees the old allocation it
waits for its own last splice, which itself waited for every peer write into
that staging, and peers cannot write the new one before they have read its
address in the rendezvous. Recordings that still point at the old staging see
the id change in their signature. Prefill makes this large: hidden size x prompt length x
element size on every rank, allocated during the first long prompt, after the KV cache took its
share.

With `TP_COMM_BUFFER_SIZE` the staging has that size, rounded down
to 512 bytes, and is allocated once by `create_context()`, before any model is
compiled; the ranks check at the first bind that they agree on it. A payload
whose staging does not fit a half is split into chunks of equal size (the last
may be shorter), each a complete allreduce with its own call number, so the
chunks alternate staging halves and event sets exactly like separate calls.
All chunks go into the one recording, separated by barriers, behind one call
rendezvous: the barrier gives a chunk the same ordering a call has after the
previous call in the in-order queue. Chunks never take halving, whose staging
is larger than the ring's, so the last, shorter chunk fits whenever the others
do. Chunking does not change a bit on two ranks; on more, the ring sums an
element in an order that depends on where the chunk boundaries fall, so the
result differs from the unchunked one in rounding, deterministically.

Three other ways to run the chunks were measured against this one: a call per chunk re-recording
the two lists (each chunk waits for the device to finish the chunk two before it, which lengthens
the time to first token), a call per chunk with a list per chunk created on first use (the
creation lands on the first prompt), and the same lists made with the group (a longer model
load, and the chunk count has to be known in advance). Mutable command lists cannot
retarget memory copies, which every transfer here is. The single recording
matched or beat all of them everywhere and costs nothing on the first prompt.

What chunking costs in itself is device time: the chunks run one after the
other, each with every step's cross-device round trip. The cost grows with the number of chunks
and of ranks; on two ranks it is negligible.

A rank's staging, grown or fixed, is counted as `usm_device` in the engine of
the rank's TP context from the first bind on, and `ov::intel_gpu::memory_statistics`
sums every live engine of a device -- default, user and TP contexts alike -- so
total device memory minus that figure is what the process has left on the
device (other processes are not seen), e.g. for sizing a KV cache.

- TP=2: one payload-sized region per half, aligned to 256 bytes.
- Ring / halving (N>2): `N` chunk slots per half, so concurrent steps never
  share a slot. Halving lays one region per scatter step and one per gather
  step over the same space, sized from the collective's own payload.

No allreduce schedule writes into a peer's model buffers: peers write staging
only, and each rank delivers into its own output. Remote writes into a buffer
the receiving rank's own kernels had just written were measured to lose against
them now and then (the halving gather used to do this, and an occasional decode run produced a
different token). The gather does write into
the root's output, but only behind a handshake with the root's queue (see
"Gather").

### Executing a collective

```
every rank
  |
  | grow own staging if the payload does not fit (allreduce, no fixed size)
  |
  | call rendezvous: publish {seq, kind, collective_id, n, dtype, rows or
  |   chunks, own staging (allreduce) or the root's output (gather)} into
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
host -- peer events, gather ready events and completion events -- so the waiting queues
drain. What they produced is garbage, so the inference that ran into it throws
once its stream has drained (`Graph::throw_if_collectives_failed()` in
`SyncInferRequest::wait()`), every later collective throws at once, and binding
another model to the group fails.

### Schedules

The schedule is chosen per recording from the world size and the payload:

| Schedule          | When                                                    | Steps       | Per-rank traffic   |
|-------------------|---------------------------------------------------------|-------------|--------------------|
| pair              | `N == 2`                                                | 1 exchange  | one payload        |
| recursive halving | `N` a power of two and payload ≤ `GPU_TP_HALVING_MAX_BYTES` | `2·log2(N)` | `2·(N-1)/N·S` |
| ring              | otherwise                                               | `2·(N-1)`   | `2·(N-1)/N·S`     |

Halving and the ring move the same bytes; what halving buys is fewer steps and
fewer round trips, which is what decode-sized payloads are made of. At prompt
sizes it loses, because both partners of a pair push across the same link at
once while the ring spreads the traffic one way around the loop — hence the
payload ceiling.

Source-side enqueue is mandatory throughout: a peer-pull memcpy silently
transfers zeros on the validated driver.

### Gather

The gather collects every rank's `rows x slice` matrix as a column band of
rank 0's output. The root publishes the output's address and allocation id in
the rendezvous, and every rank copies its band straight there: one plain copy
per row. A region copy did the same in one command, but left the first bytes
of a band row stale on the root about once in 500 calls (L3 conformance stress,
0 in 9000 with a copy per row). There is no gather staging.

A peer may write the root's output only once the root's queue has reached the
gather. Before that, the memory pool may still have the same memory under an
earlier primitive the root is running: that primitive would overwrite the bands
or take them for its own data, and dirty cache lines the root holds over the
memory could be written back over them. So, per parity:

- the root copies its own band, puts a barrier behind it and signals `ready[r]`
  for every peer; the signal has device scope, so the root's writes are
  flushed by then;
- rank `r` waits for `ready[r]`, resets it, copies its band and signals
  `ev[r]` through a barrier, so the reset is complete as well;
- the root waits for every `ev[r]` and resets them; the rest of the root's
  model is queued behind that wait.

`ready` is created unsignalled. The root signals `ready[r]` of a parity again
only from its next recording of that parity, which its in-order queue runs
after the previous one has seen `ev[r]`, i.e. after rank `r` cleared `ready[r]`.

A test that holds the root's queue back with a long copy and then writes over
the output before each gather corrupts the bands every time without `ready`.
In the LLM the gather writes the logits, a network output the pool does not
hand to anything else, which is why no model run showed it. The handshake costs
one more cross-device signal per gather; decode measured the same as with a
gather staging and as without the handshake.

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
- this rank's `RankCommunicator`, which allocates the rank's fixed staging if
  `TP_COMM_BUFFER_SIZE` is set, publishes its setup payload
  (header, process id, a process-unique communicator id, the staging size)
  and returns without waiting.

Other accepted parameters: `TP_COMM_BUFFER_SIZE`,
`COMMUNICATION_TIMEOUT_MS`, `DEVICE_ID` and the `GPU_TP_*` options.
`TP_CHANNELS` other than 1 is refused until it is implemented;
any other parameter is an error.

### Collective backends

A backend is a name the GPU plugin owns; today there is one, `L0_P2P` (peer
copies between the devices of one Level Zero context). It serves a device
group if every device appears once, all share a Level Zero driver, and
`zeDeviceCanAccessPeer` holds for every ordered pair. Two queries answer for a
device group (argument `TP_DEVICE_GROUP`) without throwing:

- `TP_COLLECTIVE_BACKENDS`: the names that serve the group, in the
  order `AUTO` prefers them; empty if none does.
- `TP_COLLECTIVE_SUPPORT` (optional argument
  `TP_COLLECTIVE_BACKEND`, default `AUTO`): empty if the backend
  serves the group, otherwise why not -- a device pair without peer access, a
  device listed twice, an unknown device, or for an unknown name the names
  that would work (`Unknown collective backend "L0"; supported for GPU.0,GPU.1:
  L0_P2P`).

`create_context()` runs the same check to resolve the group's backend (`AUTO`
becomes the first supported name) and fails with that reason. The resolved
name goes into the setup payload, and every rank checks at the first bind that
all resolved the same. It is also put into the configuration of every model
compiled or imported with the context (`GPU_TP_COLLECTIVE_BACKEND`), where the
collective implementations accept only their own backend.

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

### Device memory

`TP_DEVICE_MEMORY` (RO, argument `DEVICE_ID`) answers
`ov::tp::internal::DeviceMemory{total, used, max_allocation}` for one device:
`max_global_mem_size`, the device's memory in use by every engine of the
process (`usm_device` + `cl_mem`, plus `usm_host` on an integrated GPU) and
`max_alloc_mem_size`. The TP plugin sizes the paged-attention cache from it
without reading GPU properties.

## Plugin-managed Cache

The paged-attention cache of a TP model is split by KV head across the rank
devices, so the application cannot allocate it through one remote context.
The compiled model offers `ov::plugin_managed_cache` instead (absent on a
model without `key_cache.N` / `value_cache.N` inputs), an
`ov::PluginManagedCache` facade over the `CacheController`:

- **One KEY_VALUE group, model-level.** Tensors `key_cache.N` /
  `value_cache.N` with the shapes a single GPU reports (block dimension
  dynamic): the head dimension is the sum over the ranks with a KV head kept by
  several ranks (Case 2) counted once, and `block_size_in_bytes` is the block
  of the whole model. A block index means the same tokens on every rank; every
  operation is the same request repeated per rank and layer. Case 2 needs no
  mapping: each rank allocates from its own ports.
- **Available memory**: the minimum over the ranks of
  `(total - 1.1 * used) / rank block bytes`, capped by
  `max_allocation / largest rank tensor block`, times the model block bytes.
- **allocate / reallocate** build the complete new cache on every rank (kept
  blocks copied device-to-device), then bind it; a failure rebinds the old
  cache (or nothing) and leaves the size unchanged. **deallocate** binds
  zero-block host placeholders so the rank requests let go of the memory.
  **copy_blocks** and **zero_blocks** run on every rank tensor.
- **Binding is eager.** Every infer request registers with the controller on
  creation (unregisters on destruction) and is bound when it registers and on
  every allocation change; `infer()` only checks that a cache is allocated.
  `set_tensor` / `get_tensor` on a cache input are refused.

GenAI's continuous batching asks for the property before looking for cache
inputs and takes its memory budget from `get_available_memory()`.

## Compiled Blob

`CompiledModel::export_model` writes a container around one device-plugin
blob per rank (version 3):

```
char[8]              "OVTPBLOB"
uint32               version
uint32               world_size
uint32               strategy         (TP_STRATEGY)
uint32               num_sharded_states
uint32               state_replicas   (consecutive ranks holding the same slice of those states)
[world_size]         device name      : string
[num_sharded_states] variable id      : string
uint32 + [..]        input ports      : node name, element type, partial shape, tensor names
uint32 + [..]        output ports     : same
[world_size]         rank blob        : uint64 length + bytes
```

A string is a uint32 length + bytes. The weight shards are already baked
into each rank's blob, so no shard metadata is stored, and the device plugin
counts the collectives of each imported rank model when it binds it. The
sharded state ids and the replica count are there because the infer request has to know which
states are per-rank slices rather than replicas, and which ranks hold the same slice (fewer KV
heads than ranks): `get_state` reads one copy of each, `set_state` writes it to all of them. The
ports are the
model-level ports the compiled model reported, so an imported model reports
the same ones (rank 0's would carry its sharded cache shapes).

`Plugin::import_model` validates the header, asks the device plugin whether
it can serve the recorded devices, creates the rank contexts over them and
hands each rank blob to the device plugin with that rank's context, which
binds it exactly as after `compile_model`. The resulting `CompiledModel` has
no `ov::Model`; its ports are built from the recorded descriptions.

Everything read back is untrusted input (a stale, truncated or corrupt
cache entry), so each length is bounds-checked before it is used to size
an allocation or move the stream. A blob is bound to its topology:
importing it against a different device set is rejected.

`ov::cache_dir` works because the plugin advertises
`ov::device::capability::EXPORT_IMPORT` and answers
`ov::internal::caching_properties` — the rank device plugin's list plus the
device list (`ov::device::priorities`), `TP_STRATEGY` and
`ov::device::properties`, so two topologies, or two configurations of the
rank devices, cannot collide on one hash. A device plugin's caching property
set on `TP` is taken as it is; otherwise the device plugin's values are
aggregated across the rank devices and joined with `;`.

Weightless caching (`OPTIMIZE_SIZE`): the weight sharding pass gives a rank weight that is
a contiguous slice of an unconverted constant its own offset and size in the
model's `.bin` (`WeightlessCacheAttribute`), and drops the attribute from any
other slice, which then stays in the blob. On import the core passes
`ov::weights_path`, which the plugin supports and forwards to the ranks.

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
  per rank, created on first use and reused by every infer request of
  that model. Starting them per inference costs tens of microseconds of
  start-up skew, and that skew would be paid again at every collective.
  It is declared last so its threads are joined before anything they touch
  goes away.
- `infer_request.cpp` dispatches every rank, rank 0 included, to those
  workers and waits for all of them. The TP plugin owns the rank threads
  (option B of the design's rank execution): FIFO per rank and concurrent
  ranks follow from one worker per rank under the device locks, and the
  GPU plugin's async machinery (task executor, `EXCLUSIVE_ASYNC_REQUESTS`,
  external queues) is not on the path. Running rank 0 on the calling thread
  instead measured slower to first token on short
  prompts; a rank request on the GPU plugin's `start_async()` measured the
  same as a worker.
- `CompiledModel::lock_inference()` takes a process-wide lock of every
  device of the model, in one order (by device UUID), for the whole TP
  request: the rendezvous slots, the recorded command lists and the staging
  serve one request at a time, and TP models whose devices overlap (a main
  and a draft model) must not run their collectives concurrently, or a
  device queue waiting for a peer could hold the other model back. Rank
  execution inside the request stays parallel.
- Inside `GroupState::begin_allreduce` and `begin_gather`, ranks meet at
  a call rendezvous (a mailbox over a `std::condition_variable`, slots
  alternating by call parity). There is no record gate and no exit barrier.

## Options

Every knob is a plugin option declared in one table `include/tp/options.inl`,
where an X-macro gives each entry a typed getter, a default, validation,
and support for the environment.

The table has three tiers:

| Tier             | Reachable through       | Exists in                        |
|------------------|-------------------------|----------------------------------|
| release          | public API, environment | every build                      |
| release-internal | environment             | every build                      |
| debug            | environment             | only `-DENABLE_TP_DEBUG_CAPS=ON` |

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
under execution. `TP_VERBOSE` covers this plugin only (its timings are ITT
tasks); the collectives have `GPU_TP_VERBOSE`, `GPU_TP_PROFILING` and
`GPU_TP_DUMP_PERIOD`. The TP plugin passes no option of a device plugin down:
it does not know which device plugin runs its ranks.

For the full list see:
Public properties: [USER_GUIDE.md § Configuration Properties](USER_GUIDE.md#configuration-properties)
Dev properties: [DEVELOPER_GUIDE.md § Tunables](DEVELOPER_GUIDE.md#tunables)

## Known Limitations

- The analysis anchors on `PagedAttentionExtension` or
  `ScaledDotProductAttention` and requires the projections to be MatMuls
  over constant-derived weights. A block it recognizes only partly, or an op
  without a sharding rule on a split tensor, fails the compilation; a weight
  fed by a live parameter (a LoRA adapter) is not supported.
- Every split is even: the ranks divide the KV heads (or, with fewer KV heads
  than ranks, divide by the KV heads and divide the query heads) and the
  MLP's quantization groups.
- Remote contexts supplied by the user are not supported: the plugin
  always creates the rank contexts itself. The compiled model's own context
  gives out host tensors only; a device tensor would be one model-level
  tensor on one rank.
- Weightless caching stores only the rank weights that are not a contiguous
  slice of a weight in the model's `.bin` (slices across rows).
- `ov::num_streams` must be 1: a collective call is matched across the ranks
  by the channel of the rank model's stream, and the stream that runs a
  request is picked per request. Anything else is rejected.
- Models with linear-attention state (PagedGatedDeltaNet, PagedCausalConv1D,
  PagedSelectiveSSM, `*_state_table.*` inputs) are rejected.
- All ranks of a group run in one process.
- Every rank has to run its collectives in one order. With the collectives on
  one chain (the residual stream of a sequential block) that is given; with
  collectives on parallel branches (attention and MLP computed in parallel from one norm)
  it holds only while the device plugin rewrites every rank model
  alike. A rewrite that depends on a rank's constants (a multiplication by one
  removed on one rank) can reorder them; the GPU plugin then fails the first
  call (`ranks disagree on call ...`) instead of mixing them up.
- A failed collective is terminal. The device queues may still hold
  unfinished work, so the group refuses every later call rather than
  running on top of unknown state; the compiled model has to be rebuilt.

## Pointers

- Detailed correctness/perf debugging steps and tunables: see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).
- External API and configuration: see [USER_GUIDE.md](USER_GUIDE.md).

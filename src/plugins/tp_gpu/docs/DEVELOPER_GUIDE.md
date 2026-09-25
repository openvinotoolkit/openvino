# Tensor Parallel Plugin — Developer Guide

## Build

The plugin is **off** by default, and it needs the intel_gpu plugin built on
the Level Zero runtime rather than the default OpenCL one:

```bash
cd openvino/build
cmake -DENABLE_TP_GPU=ON -DGPU_RT_TYPE=L0 ..
make -j$(nproc) openvino_tp_gpu_plugin
```

`GPU_RT_TYPE` defaults to `OCL`, so leaving it out fails configuration outright.
`L0` is an alias for `ZE`; both work and are normalized to `ZE`.
Plugin takes the Level Zero driver and device handles from intel_gpu's remote context,
and an OpenCL-backed intel_gpu has none to give.

`ENABLE_INTEL_GPU` must be on as well. It is on by default where the GPU plugin
is supported at all, and the collective primitives are compiled into intel_gpu
itself, which links the TP ops library.

Produces `bin/intel64/Release/libopenvino_tp_gpu_plugin.so` and
registers `TP_GPU` in `plugins.xml`.

The debug option tier needs `-DENABLE_TP_GPU_DEBUG_CAPS=ON` (implied by
`-DENABLE_DEBUG_CAPS=ON`). Without it the verbosity and profiling options do
not exist at all, and the code they guard folds away.

The collective primitives live in the intel_gpu plugin and include the
coordinator's header, so **rebuild both together** whenever
`tp_device_coordinator.hpp` changes:

```bash
cmake --build . --target openvino_tp_gpu_plugin openvino_intel_gpu_plugin -j$(nproc)
```

A stale intel_gpu built against a different layout of the coordinator produces
silent garbage rather than a link error.

The plugin links against `openvino::runtime::dev`, `openvino::zero_loader`,
an OpenCL ICD (used to compile the embedded kernel into a native binary at
runtime), and `openvino::tp_gpu_ops` -- the small static library holding the
collective op definitions. That library is separate precisely because
intel_gpu links it too: its op factory casts to those types and needs their
typeinfo. The plugin also reads intel_gpu's public headers, to inject the
collective registry into each rank's compiled model.

## Source Layout

See [ARCHITECTURE.md § Source Tree](ARCHITECTURE.md#source-tree-current)
for the current file layout. Key entry points:

| File                            | Role                                                                                                                                                                             |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/plugin.cpp`                | `Plugin::compile_model` — config parse, shared L0 context creation, per-rank `GraphRewriter::rewrite` and per-rank intel_gpu compilation; `import_blob` for the cached path.     |
| `src/graph_rewriter.cpp`        | `analyze` (finds shardable MatMuls structurally and reads the model dimensions off them) and `rewrite` (per-rank weight slicing, KV-cache shape patching, collective insertion). |
| `src/tp_device_coordinator.cpp` | All L0 work: per-rank queues/cmdlists, kernel build via OpenCL→native-binary, plan caching, the three schedules, splicing and the watchdog.                                      |
| `src/infer_request.cpp`         | Rank fan-out, input replication, output collection from rank 0.                                                                                                                  |
| `src/rank_workers.hpp`          | The persistent worker thread per non-zero rank.                                                                                                                                  |
| `src/cache_controller.cpp`      | Per-rank KV-cache control inputs.                                                                                                                                                |
| `src/compiled_model.cpp`        | `export_model` / state plumbing / port derivation.                                                                                                                               |
| `src/tp_blob.hpp`               | The container format shared by export and import.                                                                                                                                |
| `src/tp_config.cpp`             | Device-name resolution and validation.                                                                                                                                           |
| `include/tp_gpu/options.inl`    | The option table: defaults, validators, tiers.                                                                                                                                   |
| intel_gpu `impls/ocl/tp_*.cpp`  | `tp_allreduce.cpp` and `tp_gather.cpp` — the OCL primitives that call into the coordinator. Registered from `graph/registry/tp_*_impls.cpp`.                                     |

## Tunables

They are plugin options, not bare environment reads: the table lives in
`include/tp_gpu/options.inl` and the X-macro there gives each entry a typed
getter, a default, validation, and environment and config-file support. An
option's environment variable is `OV_` plus its property key.

Public properties (`TP_SIZE`, `DEVICE_IDS`, `COMMUNICATION_TIMEOUT_MS`) are
documented in [USER_GUIDE.md § Configuration Properties](USER_GUIDE.md#configuration-properties).
Everything below is reachable through the environment or a config file only.

### Behaviour switches

Present in every build. They change what the plugin does, so a measurement
taken with one of them set does not describe a stock run.

| Environment variable           | Default  | Effect                                                                                                                                                                 |
|--------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OV_TP_ENABLE_HALVING`         | `true`   | Recursive halving/doubling instead of the ring for small payloads on power-of-two world sizes. Turn off to compare the two schedules on the same payload.              |
| `OV_TP_HALVING_MAX_BYTES`      | `262144` | Payload ceiling above which halving falls back to the ring. The `schedule of N recordings` line in the report says whether the current ceiling is anywhere near right. |
| `OV_TP_INPUT_STAGE_MAX_BYTES`  | `4096`   | Size ceiling for staging a user input through plugin-owned host memory. `0` always stages on the device.                                                               |

### Diagnostics

The debug tier needs `-DENABLE_TP_GPU_DEBUG_CAPS=ON` (implied by
`-DENABLE_DEBUG_CAPS=ON`). Without it those options have no getters at all,
so the guarded code folds away instead of costing a branch.

| Environment variable      | Effect                                                                                                                                                      | When to enable                                       |
|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `OV_TP_VERBOSE=LOG_INFO`  | One-line summaries per compile: sharding plan, shared context, per-rank op counts, which queue the collective rides.                                        | First thing to reach for when something looks wrong. |
| `OV_TP_VERBOSE=LOG_TRACE` | Per-call `[TP][L0]` step-by-step tracing.                                                                                                                   | Only when chasing a hang or a correctness crash.     |
| `OV_TP_PROFILING=HOST`    | Host-side measurement: rendezvous phases, per-phase splits per rank, record breakdown, scratch arena, run-ahead depth. Leaves the execution schedule alone. | The default choice when measuring.                   |
| `OV_TP_PROFILING=DEVICE`  | Device-side numbers: how long the collective owns the model queue, bytes moved, effective bandwidth, and what share of device time the collectives take.    | Only when the device-side numbers are the question.  |
|                           | Adds a kernel-timestamp query per collective, so host figures from such a run are not comparable with a HOST run -- which is why it does not print them.    |                                                      |
| `OV_TP_PROFILING=ALL`     | Both of the above.                                                                                                                                          |                                                      |
| `OV_TP_DUMP_PERIOD=N`     | Calls between two dumps. The counter is rank-0 collectives for the coordinator's report and inference calls for the dispatch block.                         | Almost always worth setting to some small number:    |
|                           | Unset means every call, which for the coordinator is one report per collective.                                                                             | by default, the output is overloaded with info       |

Verbosity and profiling are deliberately separate: raising the verbosity to see
what a run is doing should not start timing it, and asking for timings should
not require guessing which log level carries them.

### Diagnostics that break the results

Also debug-tier. Each one makes the model compute something other than what it
should; they exist to price a part of the pipeline by removing it.

| Environment variable                 | Effect                                                                                                                                                        |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OV_TP_FORCE_SYNC_COLLECTIVE=1`      | Take collectives off the spliced model queue onto their own queue with a full drain.                                                                          |
|                                      | Results stay correct; this is the one switch here that does not break them. Use it to measure what splicing buys.                                             |
| `OV_TP_DISABLE_LM_HEAD_SHARDING=1`   | Leave the vocabulary projection unsharded, dropping the gather.                                                                                               |
|                                      | Results stay correct, but the collective count baked into an exported blob changes, so blobs are not interchangeable across this setting.                     |
| `OV_TP_SHARD_ONLY=1`                 | **Wrong by construction.** Compile one rank's shard with every collective stripped, to measure the upper bound a rank could reach if communication were free. |
| `OV_TP_SKIP_COLLECTIVE=1`            | **Wrong by construction.** Return from every AllReduce without doing anything. The difference against a normal run is the whole cost of the collective.       |

Profile output anatomy (one block per dump period; unset means every call, so
the coordinator's block below appears once per collective):

```
[TP][RANK|host] calls=…  arrival spread=…us/call
[TP][RANK|host]   rank N: late=…us arrived_last=…%  ph1=…us ph2=…us  segment mean=…us min=…us max=…us
[TP][RANK|host]     ph2 split: gate=…us record=…us(Nx) evt_wait=…us evt_reset=…us(Nx, blocked Nx) append=…us rest=…us
[TP][RANK|host]     run-ahead: mean=… splices max=…  record gate: fast=…% of N
[TP][TOTAL|host] r0 calls=…  rebuilds=… records=…  totals: … per-call: …
[TP][TOTAL|host]   schedule of N recordings: pair=… ring=… halving=…
[TP][TOTAL|host]   record breakdown: records=… per-record=…ms (drain=… reset=… append=… close=…)
[TP][TOTAL|host]   scratch: payload_capacity=…MB total=…MB generation=… grows=… allocations=… stall=…ms
[TP][TOTAL|host]   gather rank N: calls=… spliced=… records=… barrier=…us append=…us close=…ms total
[TP][RANK|dev] calls=… schedule=… world=…
[TP][RANK|dev]   rank N: queue busy mean=…us max=…us(cid N)  sent=…MB/call effective=… GB/s (N samples)
[TP][RANK|dev]     queue busy under: p50=…us p90=…us p99=…us
[TP][RANK|dev]     between collectives: model=…us -> collectives own …% of device time (N gaps)
[TP][TOTAL|dev] across N ranks: sent=…MB/call aggregate=… GB/s
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
  speed of the last rank, on every collective of every token.
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
- **run-ahead** — how many splices this rank has handed over that the devices
  have not caught up with. This is what splicing buys; a mean near 1 means the
  host is not running ahead at all.
- **record gate: fast** — how often a rank could skip the gate because it could
  see for itself that neither the signature nor the arena had moved.
- **queue busy** — how long the collective occupied the rank's model
  queue, read off the completion event. This is the one part of the cost no
  host phase contains: the splice hands the work over and returns, so the
  copies, the reduce kernel and the waits on peers all happen after every host
  measurement has ended. `cid` names the collective responsible for the max.
- **between collectives** — device time spent on the model rather than on a
  collective, and the duty cycle that follows from it.
- **rebuilds** — staging arena growths since process start. After the first
  prefill of a given max-shape this should stop moving.
- **schedule** — which of the three schedules the recordings chose. `halving`
  is default-on and bounded by payload, so the split says whether that bound is
  anywhere near right. Counted per rank, so N ranks re-recording one collective
  shows up as N.
- **record breakdown** — divided by the number of recordings, not calls:
  `drain` is the queue drain before a reset, `append` the command building,
  `close` the `zeCommandListClose`. Summed over all ranks.
- **gather** — the vocabulary gather, kept out of the averages above because it
  is sized by the vocabulary rather than the hidden dimension.

## Tests

Two binaries, both under `tests/`.

### `ov_tp_gpu_unit_tests`

| Suite                                     | What it covers                                                                                                                          | Needs GPUs |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------|
| `TPGraphRewriterAnalyze`                  | Structural analysis of a synthetic transformer block                                                                                    | no         |
| `TPGraphRewriter/TPGraphRewriterSharding` | Per-rank sharding for world sizes 2/3/4, collective insertion, bias, KV-cache localization                                              | no         |
| `TPGraphRewriterLMHead`                   | Vocabulary projection split and the gather that collects it                                                                             | no         |
| `TPGraphRewriterPagedAttention`           | PagedAttention anchor and its rt_info                                                                                                   | no         |
| `TPDeviceCoordinatorTest`                 | AllReduce and gather correctness on f16/f32; plan reuse across shifting pointers, sizes and dtypes; scratch growth; in-place operation; | yes, >=2   |
|                                           | allocation churn; wider worlds; ring and halving agreeing on the same payload and the ceiling where one flips to the other;             |            |
|                                           | the spliced path matching the blocking one; timeout, abort and watchdog behaviour                                                       |            |

### `ov_tp_gpu_func_tests`

| Suite                     | What it covers                                                                                                                           |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `TPGpu/TPGpuAccuracyTest` | Сompiles the same synthetic block on one GPU and on TP_GPU, feeds both the same input and compares outputs.                              |
|                           | Parametrized over world size {2, 3, 4} and precision {f16, f32};                                                                         |
|                           | Precision is pinned on both sides, because left alone the GPU picks f16 and one ULP there is the same magnitude as a real sharding error |
| `TPGpuPagedAttentionTest` | PagedAttention path at world size 2: when a cache controller is offered at all, the geometry it reports, allocate/grow/clear,            |
|                           | block copies on every rank, refusing to infer without a cache, and prefill and two-step accuracy against a single GPU                    |
| `TPGpuSerializationTest`  | export/import round-trip, cache entries per topology                                                                                     |

The graph-rewriter suites need no GPU at all; `TPDeviceCoordinatorTest` and
every functional suite skip when the machine has fewer GPUs than the world size they ask for.

### Running them

Both binaries are built into `bin/intel64/<CONFIG>` and register with CTest
under the `TP_GPU` label:

```bash
cmake --build . --target ov_tp_gpu_unit_tests ov_tp_gpu_func_tests -j$(nproc)
ctest -L TP_GPU
```

CTest runs each binary whole, which is needed for a sanity check and
useless for chasing one failure. For that, run the executables directly and
narrow with the usual gtest flags:

```bash
cd bin/intel64/Release

./ov_tp_gpu_unit_tests --gtest_list_tests
./ov_tp_gpu_unit_tests --gtest_filter='TPDeviceCoordinatorTest.*'
./ov_tp_gpu_unit_tests --gtest_filter='TPDeviceCoordinatorTest.RingAndHalvingAgree'
./ov_tp_gpu_unit_tests --gtest_filter='TPGraphRewriter*:-*PagedAttention*'

# Parametrized suites carry the parameter in the name: world2/3/4, f16/f32.
./ov_tp_gpu_func_tests --gtest_filter='*world4*'
./ov_tp_gpu_func_tests --gtest_filter='TPGpu/TPGpuAccuracyTest.*/world2_f16'

# Chasing something intermittent:
./ov_tp_gpu_func_tests --gtest_filter='*world4*' --gtest_repeat=20 --gtest_break_on_failure
```

The two binaries are set up differently, which matters when a run fails to
start at all:

- `ov_tp_gpu_unit_tests` compiles `graph_rewriter.cpp` and
  `tp_device_coordinator.cpp` **into itself** -- the plugin is a MODULE and
  exports nothing -- so it never loads `libopenvino_tp_gpu_plugin.so` and does
  not care about `plugins.xml`. It talks to Level Zero and OpenCL directly.
  Where neither is found at configure time the coordinator suite is dropped
  from the binary rather than skipped at run time, so a missing suite here is
  a build-configuration symptom, not a machine one.
- `ov_tp_gpu_func_tests` goes through `ov::Core` and needs both
  `openvino_tp_gpu_plugin` and `openvino_intel_gpu_plugin` present and
  registered.

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
N >= 3. The ring reduce-scatter + all-gather costs `2*(N-1)/N * bytes`
per rank -- the bandwidth optimum, approaching `2*bytes` as N grows and
never concentrating on one link. Recursive halving moves exactly the same
bytes in fewer steps, which is why it wins on small payloads and loses on
large ones, where both partners of a pair contend for the same link.

## Debugging Recipes

### "The recording is redone on every call"

Symptom: `record breakdown: records=N` climbs with the call count, or the
`record gate: fast=` share is low.

A recording survives while its signature holds: the per-rank input and output
addresses, the driver's allocation ids for them, `n`, `dtype`, and the scratch
generation. Anything moving re-records everyone.

- **Addresses shift every call.** intel_gpu reallocates network buffers between
  inferences. Log `slot->in_ptrs` against `rdz.in_ptrs[rdz_set]` in the rank-0
  branch of `allreduce()` to see which one moved.
- **Allocation ids shift while addresses stay.** That is the case the ids exist
  for: a freed buffer reallocated at the same address. Re-recording is correct
  here, not waste -- it is what clears the stale residency.
- **The arena grew.** `rebuilds` in the report counts that. It should stop
  moving after the first prefill of a given maximum shape.

### "Hang on the second inference"

Almost always `zeMemFree` or `zeCommandListReset` on resources still in flight.
The coordinator guards this in three places; if you add a path that frees or
resets L0 resources, put the same guard at the top of it:

- `destroy_plan` synchronizes every rank's compute queue before destroying
  events, lists and pools.
- `record_rank` synchronizes the rank's queue before `zeCommandListReset`.
- On the spliced path the queue is not ours to drain, so `record_rank` is
  reached only after `await_previous_splice` has confirmed the previous
  handover completed.

If the hang is on the device rather than the host, the watchdog turns it into
an abort after `COMMUNICATION_TIMEOUT_MS` and names the rank and the stage.

### "A collective produces zeros or NaNs"

In order of frequency:

1. **Memcpy enqueued on the wrong rank's list.** L0 silently drops transfers
   when the source device is not the one issuing the copy. Every peer copy has
   to be recorded on the *sending* rank's list; check `record_pair_rank`,
   `record_ring_rank` and `record_halving_rank`.
2. **Cross-device wait on a non-timestamp pool.** The Intel L0 driver has
   shipped versions where an event wait across devices only resolves reliably
   when the pool carries `ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP`. The N=2 pool
   sets it unconditionally; do not remove it as part of "cleanup".
3. **An event reused without a reset.** Every event has exactly one waiter, and
   the recordings clear the ones they consumed. Adding a waiter without adding
   the matching reset leaves the event signalled for the next instance.
4. **Unaligned chunk boundaries.** The reduce kernel loads and stores through
   `vload8` / `vstore8`, which need the base pointer aligned to the vector
   width; an offset that is not corrupts a handful of elements at the head of
   a chunk. `ring_chunk` and `halving_mid` snap to `kRingAlignElems` for this
   reason, and the tail past the last whole vector is handled by a scalar loop.

### "The collective costs more than it should"

Read the report before changing anything:

- **`late` and `arrived_last` dominate.** The ranks are not arriving together,
  and the group moves at the speed of the last one. Look at `segment` -- the
  model work between collectives -- rather than at the collective itself.
- **`run-ahead` mean near 1.** The host is not running ahead, so splicing is
  buying nothing and every collective is paying its full latency. Check whether
  `TP_FORCE_SYNC_COLLECTIVE` is set or the splice extension is missing.
- **`blocked` non-zero in `evt_wait`.** The host has caught up with the devices;
  the collective is genuinely the bottleneck rather than the dispatch.
- **`queue busy` high with low `effective GB/s`.** The window includes the
  reduce kernel and the waits on peers, so a low figure usually means waiting,
  not slow links. Compare `between collectives` to see the duty cycle.
- **`schedule` shows ring where halving was expected.** Either the world size
  is not a power of two, or the payload is over `TP_HALVING_MAX_BYTES`.

## Adding a New Sharding Pattern

The rewriter is **structural**, not name-driven: `analyze()` finds every
`PagedAttentionExtension` or `ScaledDotProductAttention` in the model, then
walks backward from each attention input to the MatMul that produced it.
Friendly names are read off the ops it found, for logging and for the LM head,
but they do not drive detection. An architecture with unusual layer naming
works as-is; one with an unusual pattern does not.

To support a new pattern:

1. Extend the backward walk in `analyze()` if the projection is not a direct
   MatMul producer of an attention input -- for example when QKV is fused into
   one MatMul and split afterwards.
2. Update the shape-constant patching in `rewrite()`. In particular:
   - `local_q_heads` / `local_kv_heads` are derived assuming a
     `[B, S, num_heads, head_dim]` reshape after Q. Custom reshapes need
     custom predicates.
   - The KV-cache shape patch looks for a `Constant` feeding the state
     initializer; different state-init topologies need their own pattern.
   - PagedAttention carries its head counts in `rt_info` rather than in
     constants, which is a separate step.
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
5. Register it from `src/plugins/intel_gpu/src/graph/registry/tp_*_impls.cpp`,
   alongside the two existing ones. There is no CPU fallback to register:
   without a shared L0 context the collective cannot run at all.
6. Give both the primitive and the impl `save`/`load`. A blob-restored impl is
   not rebuilt from its node, so without them every rank comes back as rank 0.

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

The obvious wins are already taken: the collective is spliced into the model's
own queue rather than run on a private one, every rank records and submits
independently instead of serializing behind rank 0, recordings survive across
calls, and the reduce kernel is vectorized. What is left, in order of
remaining headroom:

1. **Arrival skew.** The group moves at the speed of the last rank to arrive,
   on every collective. `late`, `arrived_last` and `segment` in the host report
   measure it. The skew is built in the model work *between* collectives, so
   the lever is there rather than in the collective.
2. **Sequence Parallel.** Replace AllReduce with ReduceScatter+AllGather around
   the layer norms, which parallelizes the replicated norms as well.
   A significant rewrite in `graph_rewriter`.
3. **Recording ahead of time.** Recordings are currently laid down on the
   calling thread when the signature moves, which lands in time to first token.
   `record breakdown` prices it.

Before optimizing anything here, take a `HOST` profile and check that the
collective is actually the cost: on small models the replicated norms and the
per-call dispatch dominate, and tensor parallelism can be slower than a single
GPU outright.

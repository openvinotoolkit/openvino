# Tensor Parallel Plugin — Developer Guide

## Build

The plugin is **off** by default, and it needs the intel_gpu plugin built on
the Level Zero runtime rather than the default OpenCL one:

```bash
cd openvino/build
cmake -DENABLE_TP=ON -DGPU_RT_TYPE=L0 ..
make -j$(nproc) openvino_tp_plugin
```

`GPU_RT_TYPE` defaults to `OCL`, so leaving it out fails configuration outright.
`L0` is an alias for `ZE`; both work and are normalized to `ZE`. `COMBINED`
works too: the TP code is built into its Level Zero twin only, and the OpenCL
twin refuses a TP context.
The collectives run on Level Zero inside intel_gpu, which an OpenCL-backed
intel_gpu cannot do.

`ENABLE_INTEL_GPU` must be on as well. It is on by default where the GPU plugin
is supported at all. With `ENABLE_TP` intel_gpu compiles its TP code: the
collective primitives and the communicator under `src/runtime/ze/tp/`
(`ENABLE_TP` is defined for every target of the Level Zero stack, in
`ov_gpu_set_runtime_interface_for`). Without it none of that code is built.

Produces `bin/intel64/Release/libopenvino_tp_plugin.so` and
registers `TP` in `plugins.xml`.

The TP plugin's debug option tier needs `-DENABLE_TP_DEBUG_CAPS=ON` (implied by
`-DENABLE_DEBUG_CAPS=ON`). Without it the verbosity and profiling options do
not exist at all, and the code they guard folds away. The collectives' debug
options (`GPU_TP_*`) follow the GPU plugin's own switch,
`ENABLE_GPU_DEBUG_CAPS`.

The two plugins share no code and no headers: the TP plugin reaches the
collectives only through the private collective contract
(`src/inference/dev_api/openvino/runtime/tp/collective_contract.hpp`), so each
can be rebuilt on its own. After a change to the contract, rebuild both:

```bash
cmake --build . --target openvino_tp_plugin openvino_intel_gpu_plugin -j$(nproc)
```

The plugin links against `openvino::runtime::dev` only. The collective ops
(`ov::op::internal::TPAllReduce`, `TPGather`) live in
`src/common/transformations/include/ov_ops/tp_collective.hpp`.

## Source Layout

See [ARCHITECTURE.md § Source Tree](ARCHITECTURE.md#source-tree-current)
for the current file layout. Key entry points:

| File                            | Role                                                                                                                                                                             |
|---------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/plugin.cpp`                | `Plugin::compile_model` — config parse, rank context creation, the analysis once and the rank passes per rank (L1), per-rank intel_gpu compilation; `import_blob` for the cached path. |
| `src/transformations/`          | L1, the static library `openvino_tp_transformations`: `annotations.*` (`tp_shard` / `tp_role`, rank ranges), `attention_blocks.cpp` / `mlp_blocks.cpp` (block passes), `block_utils.*` (projection, weight-chain and shape-constant helpers), `analyzer.cpp` (`TPLayerAnalyzer`), `verifier.cpp` (`TPShardingVerifier`), `rank_passes.cpp` (`TPWeightShardingPass`, `TPInsertCollectives`). |
| `src/infer_request.cpp`         | Rank fan-out, input replication, output collection from rank 0.                                                                                                                  |
| `src/rank_workers.hpp`          | The persistent worker thread per rank.                                                                                                                                      |
| `src/cache_controller.cpp`      | `ov::plugin_managed_cache`: the paged-attention cache on every rank, request registry, available memory.                                                                         |
| `src/compiled_model.cpp`        | `export_model` / state plumbing / port derivation.                                                                                                                               |
| `src/tp_blob.hpp`               | The container format shared by export and import.                                                                                                                                |
| `src/tp_config.cpp`             | Device-name resolution and validation.                                                                                                                                           |
| `include/tp/options.inl`    | The option table: defaults, validators, tiers.                                                                                                                                   |
| intel_gpu `src/plugin/plugin.cpp` | `Plugin::create_tp_context` — the TP context from the `ov::tp::internal` parameters; `Graph::bind_collectives` in `graph.cpp` binds a model to it.                            |
| intel_gpu `src/runtime/ze/tp/`  | `rank_communicator.cpp` (publish, connect, bind, group registry), `group_state.cpp` (the group's resources: call rendezvous, staging, peer events, fallback queues, watchdog, profiling report), `collective_backend.cpp` (backend names, the `L0_P2P` support check, resolving `AUTO`), `tp_l0_shared_context.cpp`. |
| intel_gpu `impls/ocl_v2/tp/`, `impls/ocl_v2/tp_allreduce.cl` | `tp_allreduce.*`: the AllReduce impl, builds the reduction kernel with the model. `tp_allreduce_recorder.*`: the three schedules, their staging layout and the chunking of messages larger than a fixed staging. `tp_gather_recorder.*`: the gather's band copies into the root's output and its `ready` handshake. `tp_recording.*`: a primitive instance's command lists, completion events and splice. Registered from `graph/registry/tp_allreduce_impls.cpp`. |
| intel_gpu `impls/ocl_v2/tp/tp_gather.*`  | The gather impl (no kernel); runs `TPGatherRecorder`. Registered from `graph/registry/tp_gather_impls.cpp`. |

## Tunables

They are plugin options, not bare environment reads: the table lives in
`include/tp/options.inl` and the X-macro there gives each entry a typed
getter, a default, validation, and environment and config-file support. An
option's environment variable is `OV_` plus its property key.

The device list (`"TP:GPU.0,GPU.1"`), `COMMUNICATION_TIMEOUT_MS` and
`ov_internal_tp_comm_buffer_size` are
documented in [USER_GUIDE.md § Configuration Properties](USER_GUIDE.md#configuration-properties).
Everything below is reachable through the environment or a config file only.

The collectives are tuned in intel_gpu, which runs them: the `GPU_TP_*`
options, environment `OV_GPU_TP_*`. They are read when a rank context is
created. Each plugin reads only its own options: the TP plugin does not know
which device plugin runs its ranks, so it passes none of them down. To see the
whole picture, set both sides, e.g. `OV_TP_PROFILING=HOST OV_GPU_TP_PROFILING=HOST`.

### Behaviour switches

Present in every build. They change what the plugin does, so a measurement
taken with one of them set does not describe a stock run.

| Environment variable           | Default  | Effect                                                                                                                                                                 |
|--------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OV_GPU_TP_ENABLE_HALVING`     | `true`   | Recursive halving/doubling instead of the ring for small payloads on power-of-two world sizes. Turn off to compare the two schedules on the same payload.              |
| `OV_GPU_TP_HALVING_MAX_BYTES`  | `262144` | Payload ceiling above which halving falls back to the ring. The `schedule of N recordings` line in the report says whether the current ceiling is anywhere near right. |
| `OV_TP_INPUT_STAGE_MAX_BYTES`  | `4096`   | Size ceiling for copying a user input into a host tensor of every rank's context (`create_host_tensor`), which the rank then reads directly. `0` hands every input over as it is. |

### Diagnostics

The TP plugin's debug tier needs `-DENABLE_TP_DEBUG_CAPS=ON` (implied by
`-DENABLE_DEBUG_CAPS=ON`), the collectives' (`OV_GPU_TP_*`) the GPU plugin's
`ENABLE_GPU_DEBUG_CAPS`. Without them those options have no getters at all,
so the guarded code folds away instead of costing a branch.

| Environment variable          | Effect                                                                                                                                                      | When to enable                                       |
|-------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `OV_TP_VERBOSE=LOG_INFO`      | One-line summaries per compile: the analysis (blocks, geometry, collectives), rank contexts, and per rank its op count, MiB of constants and how its weights were cut (views into the original, copies and MiB copied, runtime slices). | First thing to reach for when something looks wrong. |
| `OV_GPU_TP_VERBOSE=LOG_INFO`  | The collectives: group set-up, scratch growth, which queue the collective rides.                                                                            | Together with the above.                             |
| `OV_GPU_TP_VERBOSE=LOG_TRACE` | Per-call `[TP][L0]` step-by-step tracing.                                                                                                                   | Only when chasing a hang or a correctness crash.     |
| `OV_TP_PROFILING=HOST`        | The plugin's dispatch: per-stage infer breakdown, per-rank dispatch time.                                                                                   | The default choice when measuring.                   |
| `OV_GPU_TP_PROFILING=HOST`    | Host-side measurement of the collectives: rendezvous phases, per-phase splits per rank, record breakdown, scratch arena, run-ahead depth.                    | The default choice when measuring.                   |
|                               | Leaves the execution schedule alone.                                                                                                                        |                                                      |
| `OV_GPU_TP_PROFILING=DEVICE`  | Device-side numbers: how long the collective owns the model queue, bytes moved, effective bandwidth, and what share of device time the collectives take.    | Only when the device-side numbers are the question.  |
|                               | Adds a kernel-timestamp query per collective, so host figures from such a run are not comparable with a HOST run -- which is why it does not print them.    |                                                      |
| `OV_GPU_TP_PROFILING=ALL`     | Both of the above.                                                                                                                                          |                                                      |
| `OV_TP_DUMP_PERIOD=N`         | Inference calls between two dumps of the plugin's dispatch block.                                                                                           | Almost always worth setting to some small number:    |
| `OV_GPU_TP_DUMP_PERIOD=N`     | Rank-0 collectives between two reports of the collectives. Unset means one report per collective.                                                           | by default, the output is overloaded with info       |
| `OV_GPU_TP_DUMP_PROFILING_DATA_PATH=<dir>` | With `OV_GPU_TP_PROFILING` set: `tp_collectives_<N>ranks_<k>.csv`, a row per AllReduce call and rank -- `host` rows (HOST): schedule, bytes, how late the rank arrived, rendezvous and hand-over time; `device` rows (DEVICE): the queue time of the call's splice and the model's device time since the previous collective, read when the recording is reused (the last call of each collective has none). `call` is the collective's call number, one per inference, so per-inference and per-rank sums are a group-by. | Stragglers, a step that is slower than the others, anything an average hides. HOST costs nothing measurable; ALL with the CSV +0.3 ms TPOT on TP4. |
| `OV_TP_DUMP_SHARDING_STATES=<dir>` | `compile_model` writes `tp_sharding_states_<model>_<N>ranks.txt`: the blocks found, the linear layers left whole, then every node of the analyzed model that is split or annotated with its state (Split layout per rank, Partial, rank-dependent sizes), where the state comes from, and its `tp_role` / `tp_shard`. On a verification failure the states up to the failing node, then the error. | A model fails the verification, or a new block pass is being written. |

The analyzed model and the rank models before their device compilation are not a TP option: the pass
manager serializes them, `OV_ENABLE_SERIALIZE_TRACING="TPLayerAnalyzer,TPInsertCollectives"` writes
the analyzed model with its `tp_shard` / `tp_role` attributes, then rank 0..N-1 with their collectives,
named after the models' unique names (`Model<K>`) in that order, into the working directory. Each comes
with the full weights.

Verbosity and profiling are deliberately separate: raising the verbosity to see
what a run is doing should not start timing it, and asking for timings should
not require guessing which log level carries them.

ITT tasks (`OV_ITT_SCOPED_TASK_BASE`, on in every build with the default `ENABLE_PROFILING_ITT=BASE`;
nothing measurable without a collector attached) cover the rest for VTune or any ITT collector:

| Domain | Tasks |
|---|---|
| `ov::tp::plugin` | `TP::compile_model`, `TP::check_collective_support`, `TP::create_rank_contexts`, `TP::compile_rank` (one per rank, in rank order), `TP::import_model`, `TP::export_model` |
| `ov::tp::transforms` | `TPLayerAnalyzer` > `TPBlockPasses`, `TPShardingVerifier`; `TPWeightShardingPass`, `TPInsertCollectives` (per rank) |
| `ov::tp::inference` | `TP::infer` > `TP::set_rank_inputs`, `TP::run_ranks` > `TP::rank_infer` (on each rank's worker thread), `TP::collect_outputs`; `TP::state_reset` / `state_set` / `state_get` |
| `ov::tp::collective` (GPU) | per call on the rank's thread: `tp_allreduce_rendezvous`, `tp_allreduce_wait_previous_splice`, `tp_allreduce_record` (only when it re-records), `tp_allreduce_append` or `tp_allreduce_execute_sync`; `tp_gather_rendezvous`, `tp_gather_append`; `tp_communicator_connect`, `tp_staging_grow` |

Without VTune, the ITT API's reference collector logs every task: build
`thirdparty/ittapi/ittapi/src/ittnotify_refcol/itt_refcol_impl.c` as a shared library and set
`INTEL_LIBITTNOTIFY64=<lib>` (`INTEL_LIBITTNOTIFY_LOG_DIR=<dir>` for the log).

Both sides write every message whole, so lines of concurrently logging ranks do not interleave.
`OV_TP_LOG_TO_FILE=<file>` sends the plugin's messages and reports there instead of the standard
output; the collectives follow the GPU plugin's own `OV_GPU_LOG_TO_FILE=<file>`. Give the two
different files.

### Diagnostics that break the results

Also debug-tier. Each one makes the model compute something other than what it
should; they exist to price a part of the pipeline by removing it.

| Environment variable                 | Effect                                                                                                                                                        |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OV_GPU_TP_FORCE_SYNC_COLLECTIVE=1`  | Take collectives off the spliced model queue onto their own queue with a full drain.                                                                          |
|                                      | Results stay correct; this is the one switch here that does not break them. Use it to measure what splicing buys.                                             |
| `OV_TP_DISABLE_LM_HEAD_SHARDING=1`   | Leave the vocabulary projection unsharded, dropping the gather.                                                                                               |
|                                      | Results stay correct; the setting is part of the configuration a cache entry is keyed by.                                                                    |
| `OV_TP_SHARD_ONLY=1`                 | **Wrong by construction.** Compile one rank's shard with every collective stripped, to measure the upper bound a rank could reach if communication were free. |
| `OV_GPU_TP_SKIP_COLLECTIVE=1`        | **Wrong by construction.** Return from every AllReduce without doing anything. The difference against a normal run is the whole cost of the collective.       |

Profile output anatomy (one block per dump period; unset means every call, so
the collectives' block below appears once per collective):

```
[TP][RANK|host] calls=…  arrival spread=…us/call
[TP][RANK|host]   rank N: late=…us arrived_last=…%  ph1=…us ph2=…us  segment mean=…us min=…us max=…us
[TP][RANK|host]     ph1 staging check=…us  ph2 split: record=…us(Nx) evt_wait=…us evt_reset=…us(Nx, blocked Nx) append=…us rest=…us
[TP][RANK|host]     run-ahead: mean=… splices max=…
[TP][TOTAL|host] r0 calls=…  rebuilds=… records=…  totals: … per-call: …
[TP][TOTAL|host]   schedule of N recordings: pair=… ring=… halving=…
[TP][TOTAL|host]   chunked: calls=… chunks/call=… max=…
[TP][TOTAL|host]   record breakdown: records=… per-record=…ms (drain=… reset=… append=… close=…)
[TP][TOTAL|host]   scratch: payload_capacity=…MB total=…MB generation=… grows=… allocations=… stall=…ms fixed=…MB
[TP][TOTAL|host]   gather rank N: calls=… spliced=… records=… barrier=…us append=…us close=…ms total
[TP][RANK|dev] calls=… schedule=… world=…
[TP][RANK|dev]   rank N: queue busy mean=…us max=…us(cid N)  sent=…MB/call effective=… GB/s (N samples)
[TP][RANK|dev]     queue busy under: p50=…us p90=…us p99=…us
[TP][RANK|dev]     between collectives: model=…us -> collectives own …% of device time (N gaps)
[TP][TOTAL|dev] across N ranks: sent=…MB/call aggregate=… GB/s
[TP][RANK|dev]   gather rank N: queue busy mean=…us max=…us  sent=…MB/call (N samples)
```

The tags say what the numbers are about and where they came from.
`RANK` is per rank and exists to expose imbalance between them; `TOTAL` is
rank 0's aggregate and exists to price the collective as a whole. `host` means
a host clock, `dev` means GPU kernel timestamps -- the `dev` lines appear only
under `GPU_TP_PROFILING=DEVICE` or `ALL`.

Field semantics:

- **ph1** — the call rendezvous: waiting for the other ranks to arrive. Includes
  the staging check before it.
- **ph2** — everything from leaving the rendezvous to having handed the work over.
- **late / arrived_last / arrival spread** — how far behind the first arrival
  this rank was, and how often it was the last one in. The group moves at the
  speed of the last rank, on every collective of every token.
- **segment** — model work between leaving one collective and reaching the
  next. This is where the arrival skew is built, which is why its extremes are
  kept rather than just the mean.
- **staging check** — checking, and on growth reallocating, this rank's own
  staging before the rendezvous. Near zero once the largest payload was seen.
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
- **queue busy** — how long the collective occupied the rank's model
  queue, read off the completion event. This is the one part of the cost no
  host phase contains: the splice hands the work over and returns, so the
  copies, the reduce kernel and the waits on peers all happen after every host
  measurement has ended. `cid` names the collective responsible for the max.
- **between collectives** — device time spent on the model rather than on a
  collective, and the duty cycle that follows from it.
- **rebuilds / grows** — staging growths since process start (`grows` is the
  maximum over ranks, `allocations` the sum; `payload_capacity` the minimum).
  After the first prefill of a given max-shape this should stop moving. With
  `fixed` non-zero (`ov_internal_tp_comm_buffer_size`) the staging never grows.
- **chunked** — allreduces split because the fixed staging could not hold
  them, on rank 0; only printed when there were any. Chunks run one after the
  other on the devices, so many chunks per call show up as prefill time.
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

Two binaries under `tests/`, plus the collectives' own suite in intel_gpu.

### `ov_tp_unit_tests`

| Suite                                     | What it covers                                                                                                                          | Needs GPUs |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------|
| `TPAnnotationsTest`                       | `tp_shard` / `tp_role`: set twice, serialization, rank ranges (uneven, shared, partial shares)                                         | no         |
| `TPAnalyzerTest`                          | The analysis of the synthetic blocks: roles, geometry, bias, LM head, KV-cache variables, refusals, tied embeddings                     | no         |
| `TPShardingVerifierTest`                  | The verifier's rules on hand-annotated graphs and its refusals, each with the error text                                               | no         |
| `TPRankPasses/*`                          | Rank models on 2/3/4 ranks: weights, collectives, KV-cache variables, PagedAttention head counts, fewer KV heads than ranks, fused projections | no |
| `TPAnalyzerModels`                        | Real IRs listed in `OV_TP_TEST_MODELS` (paths separated by `:`): every rank model on 2 and 4 ranks; skipped without the variable      | no         |

### `ov_gpu_unit_tests` (intel_gpu, built with `ENABLE_TP`)

| Suite                                     | What it covers                                                                                                                          | Needs GPUs |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|------------|
| `TPCollectivesTest`                 | AllReduce and gather correctness on f16/bf16/f32 with the model's `tp_allreduce.cl`; one f32 rounding per step; all ranks bitwise equal; recording reuse across shifting pointers, sizes and dtypes; scratch growth; chunking through a fixed staging; in-place operation; | yes, >=2   |
|                                           | allocation churn; wider worlds; ring and halving agreeing on the same payload and the ceiling where one flips to the other;             |            |
|                                           | the spliced path matching the blocking one; timeout, abort and watchdog behaviour                                                       |            |

### `ov_tp_func_tests`

| Suite                     | What it covers                                                                                                                           |
|---------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `TP/TPAccuracyTest` | Сompiles the same synthetic block on one GPU and on TP, feeds both the same input and compares outputs.                              |
|                           | Parametrized over world size {2, 3, 4} and precision {f16, f32};                                                                         |
|                           | Precision is pinned on both sides, because left alone the GPU picks f16 and one ULP there is the same magnitude as a real sharding error |
| `TPPagedAttentionTest` | PagedAttention path at world size 2 through `ov::plugin_managed_cache`: offered only for PA models, the model-level group,  |
|                           | allocate/reallocate/deallocate, available memory following the allocation, a failed reallocation keeping the cache, cache ports      |
|                           | refused, no inference without a cache; prefill and two-step accuracy against a single GPU (requests created before and after the  |
|                           | allocation, a reallocation in between), copy_blocks and zero_blocks checked through decode numerics                                |
| `TPSerializationTest`  | export/import round-trip, cache entries per topology                                                                                     |
| `TPContextTest`        | The rank contexts through the collective contract, without the TP plugin: two ranks reducing through their contexts, and the refusals -- |
|                           | a collective without a TP context, a collective of another rank, a second model on the channel, a context on the wrong device,           |
|                           | unsupported parameters, an unknown backend, a device listed twice; the backend and support queries; a fixed staging; device memory statistics |

The unit suites need no GPU at all; `TPCollectivesTest` and
every functional suite skip when the machine has fewer GPUs than the world size they ask for.

### Running them

The binaries are built into `bin/intel64/<CONFIG>`; the TP ones register with
CTest under the `TP` label:

```bash
cmake --build . --target ov_tp_unit_tests ov_tp_func_tests ov_gpu_unit_tests -j$(nproc)
ctest -L TP
```

CTest runs each binary whole, which is needed for a sanity check and
useless for chasing one failure. For that, run the executables directly and
narrow with the usual gtest flags:

```bash
cd bin/intel64/Release

./ov_tp_unit_tests --gtest_list_tests
./ov_tp_unit_tests --gtest_filter='TPRankPasses*:-*Fused*'
OV_TP_TEST_MODELS=/path/a.xml:/path/b.xml ./ov_tp_unit_tests --gtest_filter='TPAnalyzerModels*'
./ov_gpu_unit_tests --gtest_filter='TPCollectivesTest.*'
./ov_gpu_unit_tests --gtest_filter='TPCollectivesTest.RingAndHalvingAgree'

# Parametrized suites carry the parameter in the name: world2/3/4, f16/f32.
./ov_tp_func_tests --gtest_filter='*world4*'
./ov_tp_func_tests --gtest_filter='TP/TPAccuracyTest.*/world2_f16'

# Chasing something intermittent:
./ov_tp_func_tests --gtest_filter='*world4*' --gtest_repeat=20 --gtest_break_on_failure
```

The binaries are set up differently, which matters when a run fails to
start at all:

- `ov_tp_unit_tests` links the L1 library `openvino_tp_transformations` --
  the plugin is a MODULE and exports nothing -- so it never loads
  `libopenvino_tp_plugin.so` and does not care about `plugins.xml`.
- `ov_gpu_unit_tests` links the intel_gpu runtime statically and talks to
  Level Zero and OpenCL directly; `TPCollectivesTest` is in it only in a
  build with `ENABLE_TP` on the Level Zero runtime.
- `ov_tp_func_tests` goes through `ov::Core` and needs both
  `openvino_tp_plugin` and `openvino_intel_gpu_plugin` present and
  registered.

## Level Zero Pitfalls

### Copies must be enqueued on the source device

`zeCommandListAppendMemoryCopy` has to go on the command list of the
device that **owns the source memory**. Enqueue it on the destination
device's list and the driver reports `ZE_RESULT_SUCCESS` while silently
transferring nothing. This is why every peer copy in
`tp_allreduce_recorder.cpp` is recorded onto the sending rank's list,
never the receiving one. A collective that returns stale data with no
error anywhere is almost always this rule being violated.

### One shared context spans every device

Cross-device copies and events only work when the USM allocations and
the event pool come from a single `ze_context` created over all
participating devices (`zeContextCreateEx`). intel_gpu creates it once per
device list (`acquire_shared_context` in `tp_l0_shared_context.cpp`) and
every rank context of the group sits on it.

### Topology cost

A gather/scatter through one coordinating rank costs `2*(N-1)*bytes`
over that rank's link, so that rank becomes the bottleneck from
N >= 3. The ring reduce-scatter + all-gather costs `2*(N-1)/N * bytes`
per rank -- the bandwidth optimum, approaching `2*bytes` as N grows and
never concentrating on one link. Recursive halving moves exactly the same
bytes in fewer steps, which is why it wins on small payloads and loses on
large ones, where both partners of a pair contend for the same link.

## Debugging Recipes

### "The recording is redone on every call"

Symptom: `record breakdown: records=N` climbs with the call count.

A rank's recording survives while its signature (`TPAllReduceRecorder::Signature`) holds: its
own input and output addresses, the driver's allocation ids for them, its
kernel, `n`, `dtype`, the chunk size, and the staging addresses and id of
every rank. Each rank checks and re-records only its own.

- **Addresses shift every call.** intel_gpu reallocates network buffers between
  inferences. Compare `sig` against `m_sig[parity]` in `TPAllReduceRecorder::run()` to see
  which field moved.
- **Allocation ids shift while addresses stay.** That is the case the ids exist
  for: a freed buffer reallocated at the same address. Re-recording is correct
  here, not waste -- it is what clears the stale residency.
- **A staging grew.** `grows` in the report counts that. It should stop
  moving after the first prefill of a given maximum shape.

### "Hang on the second inference"

Almost always `zeMemFree` or `zeCommandListReset` on resources still in flight.
These places guard it; if you add a path that frees or resets L0 resources,
put the same guard at the top of it:

- `TPRecording::~TPRecording` waits for its in-flight splices before
  destroying its lists and events, and leaks them if they never finish.
- `TPRecording::reset` drains the rank's fallback queue before
  `zeCommandListReset`.
- On the spliced path the model queue is not ours to drain, so a recorder
  resets a list only after the completion event of its previous splice has
  been consumed.
- `GroupState::ensure_staging` waits for the rank's last splice before it
  frees the old staging.

If the hang is on the device rather than the host, the watchdog turns it into
an abort after `COMMUNICATION_TIMEOUT_MS` and names the rank and the stage.

### "A collective produces zeros or NaNs"

Find it first. `OV_GPU_VALIDATE_OUTPUT_BUFFER=1` (the GPU plugin's own option) checks the output of
every primitive of every rank, the `tp_allreduce` ones included, for inf / NaN and names the first
one; `OV_GPU_TP_VALIDATE_ALLREDUCE_OUTPUT=1` compares a checksum of every AllReduce output across the
ranks -- they must be bit-identical -- and names the collective and the call where they differ
(`[TP][CHECK] AllReduce output differs across ranks: <name> (collective 24, call 0): r0=0x.. r1=0x..`,
the first 20, then a count at the end: `[TP][CHECK] AllReduce outputs compared across 4 ranks: 98304
calls, 0 differ`). Both drain the queue at every check: Llama-3.1-8B TP4, 1024-token prompt, TTFT
755 ms and TPOT 9.6 ms with the checksum instead of 90 ms and 6.7 ms.

The tensors themselves come from the GPU plugin's dump, which works on every rank: with
`OV_GPU_DUMP_TENSORS_PATH=<dir>/`, `OV_GPU_DUMP_LAYER_NAMES="tpallreduce:tp_allreduce/<projection name>"`
and `OV_GPU_DUMP_ITERATIONS=<n>` every rank writes the collective's input (`src0`, its partial sum) and
output (`dst0`, the sum). The ranks' files differ by their network id, in rank order
(`program1_network1_*` is rank 0, `program3_network2_*` rank 1 on two ranks).

In order of frequency:

1. **Memcpy enqueued on the wrong rank's list.** L0 silently drops transfers
   when the source device is not the one issuing the copy. Every peer copy has
   to be recorded on the *sending* rank's list; check `record_pair`,
   `record_ring` and `record_halving` in `tp_allreduce_recorder.cpp`.
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
   reason, `pair_stride` aligns the second pair staging set, and the tail past
   the last whole vector is handled by a scalar loop.
5. **A peer copy into another rank's model buffer.** Results differ between
   identical runs, rarely (a token flips near an argmax tie). Allreduce peers
   may write staging only; the owning rank copies into its output. The gather
   is the one exception: peers write the root's output, and only after
   waiting for the root's `ready`, signalled from the root's queue at the
   gather. Check with 10+
   processes x 5 iterations on four ranks and compare output hashes; a
   difference that disappears with `TP_FORCE_SYNC_COLLECTIVE` points here.

### "The collective costs more than it should"

Read the report before changing anything:

- **`late` and `arrived_last` dominate.** The ranks are not arriving together,
  and the group moves at the speed of the last one. Look at `segment` -- the
  model work between collectives -- rather than at the collective itself.
- **`run-ahead` mean near 1.** The host is not running ahead, so splicing is
  buying nothing and every collective is paying its full latency. Check whether
  `GPU_TP_FORCE_SYNC_COLLECTIVE` is set or the splice extension is missing.
- **`blocked` non-zero in `evt_wait`.** The host has caught up with the devices;
  the collective is genuinely the bottleneck rather than the dispatch.
- **`queue busy` high with low `effective GB/s`.** The window includes the
  reduce kernel and the waits on peers, so a low figure usually means waiting,
  not slow links. Compare `between collectives` to see the duty cycle.
- **`schedule` shows ring where halving was expected.** Either the world size
  is not a power of two, or the payload is over `GPU_TP_HALVING_MAX_BYTES`.

## Adding a New Sharding Pattern

The analysis is **structural**, not name-driven: each block pass is a MatcherPass (or anchors on
an attention op and walks from it) and annotates what it recognizes; names are read only for the
log. An architecture with unusual layer naming works as-is; one with an unusual pattern needs a
block pass, and usually nothing else, because it is built from op types that already have
verifier rules.

To support a new pattern:

1. Add a block pass next to the existing ones (`attention_blocks.cpp`, `mlp_blocks.cpp`) and
   register it in `TPLayerAnalyzer`. Annotate with the helpers of `block_utils.hpp`:
   `annotate_projection` (role, weight chain, column-parallel bias), `annotate_shape_input`
   (shape constants: a split size, the lengths or bounds of a fused split), `annotate_chain`.
   Choose the segments' units so that every rank gets whole heads / groups.
2. Run the verifier on the new block (it runs inside the analyzer). If it names an op type
   without a rule, add the rule to `verifier.cpp` with a unit test of both a proof and a refusal.
   `OV_TP_DUMP_SHARDING_STATES` shows how the states travel up to the node it stops at.
3. Add the block to `BlockConfig` in `tests/common/tp_test_models.hpp`, a rank-model test in
   `tests/unit/rank_passes_test.cpp`, and a GPU accuracy test in
   `tests/functional/accuracy_test.cpp` -- the only test that shows the numbers are right.

## Adding a New TP Op

The plugin defines exactly two ops, `TPAllReduce` and `TPGather`, and
both are inserted by `TPInsertCollectives`. To add a third:

1. Declare it in `src/common/transformations/include/ov_ops/tp_collective.hpp`
   (source in `src/common/transformations/src/ov_ops/tp_collective.cpp`) and
   add it to `ov_ops/opset_private_tbl.hpp`.
2. Inherit `ov::op::internal::TPCollectiveBase`, declare with
   `OPENVINO_OP("TPMyOp", "ie_internal_opset", TPCollectiveBase)`.
3. Implement constructors, `validate_and_infer_types`,
   `clone_with_new_inputs`, `visit_attributes`. The two existing ops
   are good templates.
4. Add an OCL primitive in
   `src/plugins/intel_gpu/src/graph/impls/ocl/`. Keep the primitive POD:
   carry `collective_id` (from `ProgramBuilder::next_tp_collective_ordinal()`),
   `rank` and `world_size` as attributes and call the collective through
   `instance.get_network().get_tp_channel()`. Teach `Graph::bind_collectives`
   (intel_gpu `graph.cpp`) about the new primitive, so its rank is checked
   and it is counted when the model is bound.
5. Register it from `src/plugins/intel_gpu/src/graph/registry/tp_*_impls.cpp`,
   alongside the two existing ones. There is no CPU fallback to register:
   without a shared L0 context the collective cannot run at all.
6. Give both the primitive and the impl `save`/`load`. A blob-restored impl is
   not rebuilt from its node, so without them every rank comes back as rank 0.

## Layout Conventions

- Headers under `include/tp/` belong to this plugin alone; nothing in
  intel_gpu includes them, and this plugin includes nothing of intel_gpu.
- Internal headers (e.g. `compiled_model.hpp`, `plugin.hpp`) stay in `src/`; the L1 library
  exports its own under `src/transformations/include/tp_transformations/`.
- What the two plugins share is the private collective contract in
  `src/inference/dev_api/openvino/runtime/tp/collective_contract.hpp` and the
  ops in `ov_ops/tp_collective.hpp`. The communicator, the group state and
  the shared L0 context are internal to intel_gpu; its impls reach them
  through the network's `Channel`.

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
   A new block pass in L1 and verifier rules for the scatter and gather.
3. **Recording ahead of time.** Recordings are currently laid down on the
   calling thread when the signature moves, which lands in time to first token.
   `record breakdown` prices it.

Before optimizing anything here, take a `HOST` profile and check that the
collective is actually the cost: on small models the replicated norms and the
per-call dispatch dominate, and tensor parallelism can be slower than a single
GPU outright.

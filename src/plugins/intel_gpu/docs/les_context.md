# LES Context And Chain Discovery

P1 introduced a default-off, network-local Host task service. P2 moved
expert reads, unchanged repacking and nonblocking uploads to its worker and
separate OCL transfer lane, initially waiting for all weights before the
original MoE kernels ran. P3 adds ready-first masked GEMV execution for the
existing Decode/small-batch GEMV path; larger Prefill retains P2 scheduling.

## Configuration

- `GPU_ENABLE_LES`: internal boolean, default `false`.
- `GPU_LES_QUEUE_CAPACITY`: internal queued-task limit, default 32, range 1-1024.

These are experimental internal properties, not a public performance promise.
They cannot be passed through the public `compile_model()` property map. Use
`OV_GPU_ENABLE_LES=1` and optionally `OV_GPU_LES_QUEUE_CAPACITY` for experiments;
internal GPU tests can set the typed properties directly.
The capacity excludes the running task and is not a staging-byte budget.

## Network Ownership

[network](../src/graph/network.cpp) creates a context only for a non-internal
network with LES enabled, a nonzero offload ratio and an executable compressed
MoE primitive with a nonzero OTD cache capacity. Disabled LES, resident-only
and non-MoE graphs create no context or worker.

An eligible network requires an offload ratio of 1-99, exactly one configured
stream and record/replay disabled. Unsupported combinations fail construction;
they do not silently fall back to a different scheduling mode. Each network
owns its context, including networks instantiated from the same program.

Each `execute()` starts a new epoch, drains Host tasks before returning and
cancels them if execution or the worker fails. Destruction stops and joins the
worker before releasing network resources. P2 callbacks drain their issued DMA
before returning, including failure paths. There is no new public infer-request
cancellation interface.

## Worker Contract

[les_context](../include/intel_gpu/runtime/les_context.hpp) owns one worker and
a bounded FIFO. The service is independent of OCL/ZE handles and per-layer
expert-weight providers. It is not a process singleton.

- `begin_epoch()` cancels the old queue, waits for running Host work to finish,
  clears the previous epoch's error and returns a new epoch identifier.
- `submit(epoch, work)` blocks while the queue is full. It returns an epoch-tagged
  shared completion future. Empty work and stale epochs are rejected.
- `wait()` waits for Host work and its captured resources to finish, then
  rethrows the worker error, if any.
- A task exception fails its ticket and all queued tickets. Blocked producers
  wake and receive the same error. A new epoch permits recovery.
- `cancel()` invalidates the epoch, cancels queued work and waits for active
  work. It does not preempt a running read or a previously submitted GPU command.
- `stop()` additionally rejects new tasks and joins the worker. Repeated stop
  is harmless. Destruction calls stop.

Control operations and submit must not run inside a worker task: they can
otherwise wait for the worker itself. The service detects and rejects that
reentrancy. Control transitions are serialized; producers must retain the
context's owner until they have returned. Stop joins the worker, not external
producer threads.

Cancelled/failed queued closures are destroyed outside the task mutex. No GPU
wait or worker join is performed while holding that mutex.

## Host Completion Is Not Device Completion

A ticket means the Host callback has returned and released its captures. It
does not imply an upload event has completed. Epochs invalidate stale metadata;
they do not fence DMA or permit slot/staging reuse. P2 separately retains
staging and native events until uploads and previous consumers complete.

## Validation

The existing property fixture covers defaults, clone/roundtrip and invalid
capacity. [Lifecycle tests](../tests/unit/test_cases/moe_les_context_test.cpp)
use fake Host work, without a GPU, to cover single-worker FIFO, queue backpressure,
cancellation, error fan-out/recovery, epoch rotation, blocked producers,
reentrant control, repeated destruction, active-task join and captured-resource
release. The same source can be built with ThreadSanitizer.

The existing MoE numerical fixture also checks LES-on resident and OTD cases,
independent contexts, repeated execute epochs and rejected stream/ratio/recording
combinations. The property fixture checks that a non-MoE graph with LES enabled
and ratio 0 or 50 creates no context and preserves its output.

## Verified Milestones

| Stage / Step | Commit | Verification |
|---|---|---|
| Stage 1 / P1.1 | `580cb907ba` | Default-off configuration; 184 focused OCL tests passed |
| Stage 1 / P1.2 | `736dc82af2` | 12 worker tests, the same 12 under TSan, 17 integrated configuration/lifecycle tests passed |
| Stage 1 / P1.3 | `903156e707` | 26 P1-focused tests, 205 complete focused tests, 8 LES-on forced-fallback tests passed |
| Stage 1 / P2.1a | `a75f2d0b7e` | 12 cross-queue event tests and 215 complete focused OCL tests passed |
| Stage 1 / P2.1b | `be2210dcaa` | 12 CPU worker tests and 27 integrated LES/lane tests passed |
| Stage 1 / P2.2 | `e5a06299d5` | 31 reservation/LRU/provider tests passed |
| Stage 1 / P2.3 | `0b7014a295` | 14 upload/provider tests passed |
| Stage 1 / P2.4 | `9a8dcff850` | 303 complete focused tests, 80 LES-on forced-fallback tests, source Qwen ratio-80 LES-on smoke passed |
| Stage 1 / P3.1a | `94d1eaae6d` | Canonical pair snapshots and nonblocking demand foundation; 307 focused tests and 80 forced-fallback cases passed |
| Stage 1 / P3.1b-P3.3 | `000d056bf2` | Masked GEMV, ready-first demand overlap, bounded multi-wave execution; 331 focused tests, 104 forced-fallback cases, source model and actual overlap trace passed |
| Stage 2 / P4.1a | `02ec1d5e1c` | Read-only static MoE chain candidates; 16 targeted integration, 339 focused and 104 forced-fallback tests passed |
| Stage 2 / P4.1b | `0555339301` | Qualify constant single-token Router/FC descriptors; 345 focused and 104 forced-fallback tests passed |
| Stage 2 / P4.2-P4.3 | `64ed628a5a` | Opt-in independent proxy Top-k, recall diagnostics and worker prefetch; 354 focused and 112 forced-fallback cases passed |

P1.3 validation used the source Debug/OCL build on Intel Arc Lunar Lake with
assertions and existing numerical tolerances unchanged. All listed runs had
zero failures and skips; the subsets overlap and are not additive. The fallback
run disabled both `OV_GPU_MOE_USE_GROUPED_GEMM_PREFILL` and
`OV_GPU_MOE_USE_MICRO_GEMM_PREFILL`. The complete focused run is not the full GPU
unit suite, and no new ZE LES or real-model performance claim is made.

## P2.1a: Strict OCL External Dependencies

The opt-in `ocl_stream::enqueue_external_barrier(deps)` queues a device barrier
for individual native OpenCL events, independently of profiling and the stream's
normal synchronization mode. Unlike the legacy dependency conversion, it rejects
null dependencies, missing native handles, grouped events, foreign contexts and
events already known to have failed. All dependencies are validated before the
barrier is enqueued. An empty list is a no-op, not an all-queue wait.

Callers must explicitly flush producer queues so their events can progress.
Returning from this method means only that the barrier was enqueued. Subsequent
commands cannot consume the uploaded data before its dependencies complete;
execution errors arising later must still be observed through a real completion
event. There is no Host wait or compute-queue finish in this method.

Grouped events are rejected because their representative native handle does not
prove completion across independent producer queues. P2 upload aggregation must
provide the individual weight, scale and zero-point completion events instead.
The caller remains responsible for source-memory, staging and slot lifetimes.

P2.4 production LES demand uses this API with every returned upload event.
The default LES-off execution path remains unchanged. ZE adaptation is not
implemented.

Validation on the source Debug/OCL Lunar Lake build passed 12 cross-queue event
tests, including the two original native controls, and 215 complete focused
tests. Both runs had zero failures and skips; the subsets overlap. The external
visibility checks split the input between two independently gated producer
queues and use both in-order and out-of-order compute queues, with profiling
on and off. They also cover missing handles, grouped native events, context
mismatch, known errors and errors arising after barrier enqueue. No sleeps or
queue finish are used to establish visibility. This is not the full GPU suite,
ZE validation or a performance/overlap result.

## P2.1b: Transfer-Lane Ownership

Each eligible network creates a separate in-order transfer stream owned by its
LES context. The context remains backend-neutral; its CPU-only tests do not
require GPU headers or libraries. Destruction joins the worker before releasing
the stream. A GPU task must drain every upload it issues before returning or
unwinding, so stop/join also covers its staging and device work. The generic Host
task API itself cannot infer outstanding DMA from a callback.

The current asynchronous LES backend is explicitly OCL-only. LES-on eligible ZE
networks fail construction rather than using unverified event semantics; LES-off
and resident/non-MoE networks retain the existing path. A gated real-upload test
checks that context destruction waits for active device work, retains the stream
and source data, and releases the stream only after completion. The 27 integrated
tests also include the existing LES-on numerical checks. No dedicated copy-engine
selection or overlap performance is implied by creating a separate queue.

## P2.2: Slot Reservations

`ExpertSlotReservations` is separate from the legacy synchronous LRU. It checks
the entire deduplicated demand before changing state, pins every hit before
assigning misses and returns immediately when capacity or active consumers block
the request. Reservation does no I/O. Tickets carry epoch/id and internal slot
generations; late completion and duplicate release cannot affect a reused slot.
Only explicit completion marks reserved contents ready. Releasing an incomplete
ticket discards its pending entries, while ready entries remain cached.

Waiting for upload/consumer events must happen outside the reservation mutex;
the caller must drain old DMA before releasing its ticket. Five new tests plus
26 existing LRU/provider tests passed. P2.4 integrates this state machine with
upload completion and real consumer events; metadata alone is not a GPU fence.

## P2.3: Bounded Staging And Upload Completion

`fill_weights_memory_async` is a worker-side pipeline, not a nonblocking Host
return: it uses nonblocking copies while reading/repacking subsequent data, then
drains every issued upload before returning or propagating an error. Its returned
events cover every uploaded weight, scale and zero-point component. The worker
must open/use its positional reader itself; execution threads do not lazy-open it.

The entire read/slot plan is checked before the first copy. Two reusable 8 MiB
staging buffers chunk large weight tensors; a buffer is reused only after its
previous event completes. Transposed scale/ZP components must fit 8 MiB each per
expert. Including the existing packed-u4 transpose temporaries, transient staging
is bounded conservatively by 64 MiB per active worker. Oversized metadata is
rejected explicitly; the legacy synchronous path is unchanged.

Six upload tests and eight existing provider tests passed: all nine components
with profiling on/off, source preservation across staging reuse, preflight
validation, missing files, short reads and a later short read after an earlier
DMA has already been submitted. LES worker/chunk/byte counters are separate from
legacy synchronous tensor-load and timing counters. No device performance claim
is inferred from these counters.

## P2.4: Production Demand Integration

The offload provider reserves the complete demand before dispatching work. Its
worker owns positional file reads and staging; all nine W/S/ZP components and
every chunk must complete before the compute barrier and GPUReady transition.
A real compute-queue marker retains the last reservation until its consumer
finishes, independently of profiling. Consumer waits occur outside reservation
locks. Failed or cancelled queued demands release incomplete reservations;
active tasks drain uploads before cleanup and the next epoch can recover.

LES clones each provider's mutable reader/cache state per network and allocates
network-private, per-layer OTD buffers. The buffers use the existing partial
physical-slot/full-logical-layout convention. Original compiled constant
allocations still exist: private buffers add persistent memory beyond the
64 MiB transient staging bound. This is not a cross-layer pool or an unchanged
memory-footprint claim. Requests exceeding capacity use the original per-expert
fallback; no split arithmetic, changed tolerances or removed legacy waits are
introduced.

Final Debug/OCL validation on Lunar Lake passed 303 tests in 25 suites and
80 LES-on forced-fallback cases, with zero failures/skips. The overlapping
subsets include 18 upload/provider cases and 80 LES-on numerical cases spanning
U4/U8, 1/4/48 tokens, 1/2/4 slots, profiling on/off, both routing modes, changing
routes, independent networks and missing-file recovery. Gated native upload,
consumer and queued-cancellation tests exercise lifetime and retry behavior.
This is not the full GPU unit suite. A fresh standalone CPU worker TSan run
could not start (`unexpected memory mapping`); the earlier P1 TSan result is
historical, not a P2 provider/driver race-safety result.

The source Qwen3.6-35B VLM smoke passed with ratio 80, LES enabled, one stream,
21 input tokens and 8 generated tokens. Core, GPU/CPU plugins, IR frontend,
GenAI and tokenizers were checked against their source-build paths. Counters
recorded 2,930 worker tasks, 29,349 upload chunks, 5,329,465,344 upload bytes and
zero legacy synchronous tensor loads. Experiment evidence remains outside this
repository in the workspace `les/results/p2_source_les_on_ratio80_env/` directory.
The first attempt preserved in a separate directory failed because the
experimental entry point passed an internal option through the public API;
the successful run uses its supported environment variable. This single short
smoke is not model-accuracy, throughput, overlap or speedup evidence. Eligible
LES-on ZE execution remains explicitly unsupported.

## P3.1a: Canonical Pairs And Nonblocking Demand

Reservation tickets now snapshot `ready_pairs` and `missing_pairs` while holding
the metadata lock. Indices are original routed `token * top_k + rank` positions,
not compacted ranks, unique experts or shared-expert positions. Both lists keep
canonical order and cover the entire request, including repeated experts across
tokens. A pair belongs to the ready list only if its slot was already ready at
reservation time; later completion does not rewrite this historical snapshot.
Hits are pinned before any miss receives a slot, as in P2.

`begin_les_demand` reserves and dispatches missing uploads, then returns slot and
pair snapshots without waiting for uploads. It may still wait for previous
consumers or bounded worker queue capacity; it is not a generally wait-free API.
`finish_les_demand` observes worker errors, drains all uploads through the worker
contract, enqueues the strict native-event barrier and marks the ticket ready.
It must be called once for a successful demand. A second finish is rejected.
Capacity rejection returns no demand and dispatches no work. The P2 synchronous
acquisition is now a begin/finish compatibility wrapper.

Releasing an unfinished demand waits for its worker before releasing pins;
its pending cache entries are discarded rather than committed. Error cleanup
also releases resources after waiting and propagates the worker error. Provider
destruction waits for an outstanding callback before destroying reader/config
state captured through the provider. Explicit finish, context wait or release
remain responsible for reporting worker errors; destruction does not throw.
Consumers must still be fenced by the caller before a reservation is released.

Validation exhausts all 256 Top-8 cold/hot masks using two differently ordered
tokens with duplicate experts. Three native-gated provider tests check begin
returning before upload completion, preserved hit data and canonical mapping,
finish blocking, repeated-finish rejection, unfinished release/retry and active
upload destruction. Final Debug/OCL evidence: 24 reservation/LRU tests,
21 upload/provider tests, 307 complete focused tests and 80 LES-on forced-fallback
cases passed with zero failures/skips. Subsets overlap. A test-fixture file/layout
size mismatch was corrected without weakening production range validation.

At the P3.1a milestone, production kernels still used the P2 all-ready wrapper.
Its 307-test and P2 real-model evidence must not be interpreted as validation
of split GPU execution. The remaining P3 integration is described below.

## P3.1b-P3.3: Ready-First GEMV And Multi-Wave Execution

LES-off GEMV retains its original kernels and arguments. Eligible LES-on GEMV
uses separate masked gate/up and down variants. A zero mask makes the entire
workgroup return before reading weights or scratch; selected workgroups keep
the original arithmetic and write the original token/rank scratch positions.
The original reduction runs exactly once after all routed pairs finish. Shared
expert work is selected only in the first wave's Pass A, once per token.

After the real routing dependencies complete, the Host reads Top-k without the
old unconditional compute-queue finish. A wave selects at most the slot capacity
in unique experts and includes every still-pending pair of each selected expert.
Reservation pins hits before assigning misses; missing reads/repacking/uploads
start on the worker. Pass A runs the already-ready pairs while this work proceeds.
Finish observes upload completion and installs the strict barrier; Pass B runs
only missing pairs. Both passes write canonical positions, independent of order.

Each wave requests a real kernel completion event, flushes compute and waits for
that consumer before releasing temporary slot/mask buffers or reusing slots.
This is a narrow event wait, not a steady-state whole-queue finish. All-cold or
all-hot demands skip the empty routed pass. Capacity-one execution makes forward
progress through bounded waves rather than switching to a different accumulation
kernel. Exceptions drain compute and worker activity before releasing buffers;
cleanup retains the original error. The initial per-network ownership finish
and legacy larger-Prefill waits remain; this is not a claim that every finish
was removed from the plugin.

Final Debug/OCL verification on Lunar Lake passed 331 tests in 27 suites and
104 LES-on forced-fallback cases, with zero failures/skips. The eight Top-8 GPU
cases each exhaust all 256 masks with U4/U8, 1/4 tokens and both routing modes;
outputs are compared bitwise to the original unsplit GEMV. Existing LES matrix
cases also compare bitwise for GEMV-sized inputs, including 1/2/4-slot multi-wave
execution and profiling off/on. Sixteen shared-expert cases compare bitwise with
unsplit shared GEMV across U4/U8, 1/4 tokens, 1/4 slots and profiling off/on.
Mixed mask 0x55 additionally injects a short read after warm-up, then checks
next-epoch recovery bitwise. Subsets overlap; this is not the full GPU suite.
The shared fixture's router layout was corrected to [batch, seq, experts] after
its old four-dimensional layout produced only one token of Top-k. No production
range guard or numerical tolerance was weakened.

`MOE_LES_TRACE=1` enables opt-in JSON interval logging in debug-capable builds
with `OV_VERBOSE=1`. Both queues must have profiling enabled. Positional-read
intervals sample the OpenCL device timer before/after the read; Pass A/B kernels
use native profiling timestamps on that same device. Missing timer/profiling
support is an error, not a fabricated Host-enqueue interval. Trace is off by
default and perturbs timings; reads may be served from page cache. It demonstrates
read/compute concurrency, not cold-SSD DMA or dedicated-copy-engine throughput.

The trace GPU probe observed 148 read/Pass-A interval intersections. Actual source
Qwen ratio-80, 21-input/8-output-token generation passed and observed 437 such
intersections, with no synthetic delay. Its counters recorded 361 waves, 1,659
ready pairs, 7,301 missing pairs, 29,457 uploads and zero legacy tensor loads.
Path/hash provenance, trace and generation reports remain outside this repository
under workspace `les/results/p3_source_les_trace_ratio80/`.

Matched trace-off source runs used ratio 80, 8 output tokens, one warm-up and
three measured runs per LES setting. All output text matched. Mean wall generation
was 3.330858 s off versus 2.534071 s on; mean TTFT was 2.536180 s versus 1.427640 s,
but mean TPOT regressed from 0.113407 s to 0.157922 s. These short sequential Debug
runs have uncontrolled page cache and are not a general speedup guarantee. LES
remains default-off; Decode overhead needs performance tuning. ZE remains
unsupported, larger-Prefill split and cross-layer prediction/pooling are not P3,
and the previously documented TSan environment limitation is unchanged.

## P4.1a: Static Chain Candidates Only

`program::get_les_moe_links()` queries the program dependency graph and returns
value descriptors containing source/target MoE IDs, the source's hidden-input ID
(`proxy_hidden`), the target's hidden-input ID, and a status. It does not infer
causality from execution order or node names. LES-off and internal programs
return no descriptors. Callers must query the completed graph; results are
snapshots, not references to mutable graph nodes or live runtime bindings.

Discovery starts at each MoE's hidden input, not its routing or weight inputs.
It traverses ordinary dependencies, stops at MoEs and constant data, and rejects
loop/condition boundaries. It removes older MoE ancestors exposed by residual
bypasses before testing uniqueness. Multiple independent nearest predecessors
are ambiguous; a predecessor reaching multiple next MoEs is branched, including
when another successor is ambiguous. Neither case is a usable chain candidate.
Missing hidden dependencies are explicitly unsupported, and a chain root has
`no_predecessor`. IDs are returned in deterministic target-ID order, not layer
execution order. No graph edges, output tensors, memory plans or cache bytes
are changed. The query stores no graph pointers and runs only when requested.

`candidate` means topological adjacency only, not permission to launch prediction.
It does not validate Router weights/bias/routing semantics, OTD eligibility,
single-token shapes or hidden-buffer lifetime. No runtime consumer, proxy-router
kernel or speculative I/O is enabled by this step. Unsupported graphs remain on
the existing asynchronous-demand path. The current algorithm conservatively
rejects branches rather than guessing a chain, and may return no useful candidate
on more complex models.

Eight graph tests cover ordinary-layer/residual gaps, nearest-predecessor
selection, independent chains, merges, forks, Router-side dependencies, a loop
boundary, missing hidden inputs, LES-off and non-MoE graphs. Existing real MoE
fixtures also query their optimized program and retain all original numerical
and bitwise checks. Final Debug/OCL verification passed 339 tests in 28 suites
and the overlapping 104-case forced-fallback subset, zero failures/skips; the
16-case targeted run is included, not additive. The plugin was relinked. The
residual probe first failed as intended and passed after ancestor filtering.

P4 proxy execution is implemented for the documented plain-FP16, single-token
projection boundary. It is not a general predictor for arbitrary Router graphs:
compressed/quantized FC, fused FC operations, unsupported layouts and malformed
chains remain demand-only. Router descriptors are re-derived from the restored
program graph rather than serialized. A two-layer synthetic supported proxy
program was saved and loaded, then rerun with the restored proxy stage; descriptors,
prediction, actual route and bitwise output checks passed in all proxy variants.
OTD partial constants now serialize an explicit physical-byte marker with no
device readback; import reconstructs a resident-sized physical backing and full
logical view. Qwen cold export/warm import parity also passed in demand-only mode.
Formal performance benefit on a model with a supported proxy chain remains
unverified. Full-suite, ZE and TSan limits remain unchanged.

P4.1b adds `program::get_les_router_links()` as a stricter, read-only
qualification query. It accepts only a matching Router output-0/1 pair,
Softmax or SigmoidBias routing, a plain noncompressed/nonfused transposed FC
projection, constant data parameters, and static single-token proxy/target
hidden layouts. It returns IDs for the Router, FC, weights and optional
projection/routing bias and epsilon. It verifies descriptor input ports,
dimensions and element types. Unsupported shapes, variable parameters, and
malformed Router/projection graphs receive explicit statuses; no parameters are
copied, no kernel launched, and no existing route modified. Qualification is not
cached or serialized and must be queried from the final compiled program.

Fourteen targeted graph cases cover chain topology plus both routing modes,
optional FC bias, nonconstant weights, incorrect Router ports, unrelated Top-k
producers and multi-token rejection. Debug/OCL verification passed 345 focused
tests and 104 forced-fallback tests, zero failures/skips; the plugin linked.

## P4.2-P4.3: Independent Prediction And Optional Prefetch

`OV_GPU_LES_PREDICT=1` opts in to a plain-FP16 proxy FC and the existing Router
Top-k kernel. The source MoE copies its input activation to network-owned memory
with the upstream dependency event; the snapshot survives activation reuse and
is cleared at each LES epoch. The proxy uses dedicated logits, indices and
weights buffers. It never writes actual Router outputs or changes actual Top-k,
routing weights or MoE compute. Router descriptors are derived from the loaded
program graph; the proxy kernel stage is restored through the ordinary primitive
implementation cache path. Data cache records use the separate partial-OTD marker
documented above.

`OV_GPU_LES_PREDICT_COMPARE=1` waits for proxy and real Router IDs and records
Top-k slot recall. This mode is diagnostic and synchronizes Host readback; it is
off by default. `OV_GPU_LES_PREFETCH=1` separately opts into worker-side loading
for predicted experts. It uses the bounded LES worker, reservation and transfer
lane, skips demands that cannot fit, releases speculative pins after upload, and
lets real demand reserve and validate the actual experts. A speculative read or
upload failure drains issued DMA, discards the incomplete reservation and does
not poison the request epoch; real demand retries via the normal path.
Unsupported graphs never build a prediction stage and remain demand-only.

Eight GPU variants cover Softmax/SigmoidBias, FC bias on/off, prediction-only and
prefetch. The prediction tests observe 0/8, 1/8, 3/8 and 8/8 Top-k matches. They
check valid IDs, preserve actual Router output and compare final target outputs
bitwise with LES-off. Repeated execution verifies snapshot/event/ID refresh.
Provider tests inject a speculative short read, check that the epoch remains
usable, then confirm demand loads the repaired file. Final Debug/OCL verification
passed 354 focused tests in 29 suites, 112 overlapping LES-on fallback cases and
all eight proxy variants, zero failures/skips. The plugin linked. In the prefetch
variant the complete program was serialized and loaded into a new network; restored
descriptors and proxy stage executed, and the final output remained bitwise equal
to the LES-off baseline.

Source Qwen ratio 80 generated successfully with prediction and prefetch options
set. All 39 adjacent chains were rejected because the Router FC uses compressed
weights; zero proxy comparisons ran. This confirms conservative demand-only
fallback, not prediction coverage on Qwen. No formal speedup for a supported
model is claimed; speculative prefetch remains opt-in and its latency benefit is
unestablished.

For compiled-model cache validation, `LES_CACHE_DIR` is optional in
`run_source_baseline.sh` and is forwarded as OpenVINO `CACHE_DIR`. Qwen ratio 80
with LES enabled, prediction/prefetch requested and a fresh cache completed a
cold export and a second warm import; both generated the same text and 21/2 input/
output token counts. The cache occupied 2.2 GiB. Since this Qwen graph has no
supported proxy chain, this validates OTD partial-constant serialization and
demand-only imported execution, not imported proxy-stage execution. The dedicated
`cache_serialization` suite passed 10/10, including the new zero-payload OTD
partial allocation roundtrip. Cache timing is not a performance result.

## P4.4: Eligible Real-Model Functional Acceptance

Qualification accepts an unfused reshape data-edge chain only when it resolves
to the exact target hidden node/output. Reshape cannot authorize a different
hidden source or an intervening computation. Proxy, target and FC hidden layouts
must retain a static feature width matching the FP16 constant matrix. Leading
dimensions may be dynamic only if they admit one token. Static multi-token and
dynamic-width inputs remain unsupported. At execution, snapshots are taken only
from contiguous FP16/bfyx single-token buffers; a multi-token Router does not
launch proxy kernels. Prefill continues through the original demand path.

The workspace experiment `les/prepare_p4_router_ir.py` materialized 40 Router
matrices of the existing Qwen IR as FP16, leaving compressed expert payloads
unchanged. It requires exact equality to the original represented weights; it
does not restore their pre-quantization values or implement compressed-FC proxy
execution. Product compressed/fused FC guards remain unchanged.

On the OCL Debug host, `les/run_p4_acceptance.py` passed six sequential modes:
LES-off, demand-only, prediction, prefetch, cold compiled cache and warm reuse.
Each used one warm-up plus two measured generations of eight tokens. All 18
generated texts and token counts agreed. Each enabled prediction mode qualified
39 chains, compared 6,552 expert slots and matched 2,133 slots. Prefetch modes
recorded 6,552 prefetched slots and zero failures. Counts correspond to decode
only; the 21-token prefill did not run predictions. Warm reuse retained the five
cache blobs unchanged and still executed predictions. Experiment evidence is in
workspace `les/results/p4_acceptance_20261102/acceptance.json`.

Focused GPU tests passed 357/357, and the overlapping forced-fallback subset
passed 112/112 with grouped/micro prefill explicitly disabled. New boundaries
cover same-source reshape acceptance, different-source rejection, dynamic tokens
with fixed width and unknown-width rejection. Eight proxy variants retain strict
bitwise LES-off final-output checks, including repeated execution and synthetic
program import. Real-model text/count parity is not logits-bitwise accuracy.

This closes the eligible-real-model P4 functional gate, not a performance gate.
Measured Debug generation means were 2.014619 s demand-only, 2.111490 s prediction
and 2.404974 s prefetch: speculation regressed, rather than demonstrating speedup.
Page cache was uncontrolled and diagnostic comparison readback was enabled.
Prefetch remains blocking at the target Router; no P5 early scheduling, P6/P7
support, general GPU suite, ZE LES or new TSan result is claimed.

## P5.1: Demand-Priority Worker Foundation

`les_context::submit` remains the required-demand API. Demand tasks retain FIFO
order within their class and run before queued speculative tasks. If the bounded
pending queue is full, demand may displace a not-yet-started speculative task;
its ticket completes with `les_task_cancelled`. Already active work is not
preempted, and its uploads still drain before buffers can be reused.

`try_submit_speculative` never waits for queue space. It returns an empty optional
when the queue is full or invalidating, while stale/stopped epochs remain errors.
Speculative failures complete their own ticket without poisoning required work.
Required-demand failures preserve the previous context-wide error propagation.
Epoch rotation, cancellation, shutdown, queue capacity and the single worker
contract are unchanged. Prefetch now uses the speculative API, releases its
reservation on admission refusal and treats policy cancellation as a skip rather
than an upload failure.

Focused OCL Debug tests passed 396/396 with zero failures/skips. New deterministic
tests cover demand-before-speculation, per-class FIFO, full-queue displacement,
nonblocking prefetch refusal, speculative error isolation and stale-epoch drops.
The provider refusal test checks unchanged GPU bytes and a successful subsequent
real demand. Evidence: workspace `les/p5_foundation_results.xml`.

OLMoE six-mode regression also passed with all 18 texts/counts equal, 15 qualified
chains, 2,520 compared slots per prediction mode, 2,520 prefetched slots per
prefetch mode, zero prefetch failures and unchanged warm cache blobs. Evidence:
workspace `les/results/p5_1_olmoe_acceptance_20261102/acceptance.json`.

At the P5.1 checkpoint this was only the scheduling foundation: prefetch still
blocked at the target Router and rejection did not restore replaced mappings.
The remaining P5 implementation and validation follow below.

## P5: Early, Bounded Speculation And Pollution Accounting

The source MoE snapshots its qualified single-token hidden input and invokes the
next Router's proxy stage before submitting source expert compute. The proxy uses
fixed single-token scratch, not the unexecuted target's dynamic input/output
buffers. Actual routing and expert arithmetic are unchanged. The Router reuses
the existing proxy result rather than predicting twice. The small prediction and
ID readback still have a Host wait; weight reads/repacking/uploads run on the
worker and no longer block the source until their completion. This enables
overlap but is not a new measured device-overlap or speedup claim.

`LesSpeculationPolicy` separates `lookahead_layers=1` from `predict_k=4`.
Each provider keeps at most one speculative batch. Its admission limit is the
minimum of four experts, `max(1, capacity / 4)` slots and 16 MiB divided by the
complete per-expert W/S/ZP byte size. Capacity one, zero-byte metadata and experts
larger than 16 MiB disable speculation. These are fixed internal limits, not new
public properties or a cross-layer memory budget. No adaptive or trained
predictor is required.

Matching pending expert requests reuse the original task. A different request
does not create another batch. Real demand promotes a matching queued task to
required FIFO priority instead of uploading again, or cancels an unneeded queued
task. Active uploads cannot be preempted; demand may wait for that bounded batch.
Epoch cancellation drains active DMA, drops pending work and allows retry.

Speculation may replace only unpinned LRU slots; real active reservations are
never victims. Predicted hits do not refresh demand recency. A task rejected or
cancelled before any upload rolls back replaced metadata because device bytes
are still unchanged. Once any DMA has occurred, rollback is forbidden: cleanup
waits for all issued transfers, invalidates unused predictions and releases pins.
Useful predicted slots remain ready for actual demand. This is bounded soft
protection, not a guarantee of zero cache pollution. Independent demand-only
shadow reservations track the same successful demand sequence and capacity,
without prediction or any device allocations.

Counters are cumulative and require `MOE_OTD_PERF_LOG=1`:

- `les_early_predictions`: proxy predictions created from the source hook.
- `les_prefetch_experts`: admitted missing experts, not proof of completed DMA.
- `les_prefetch_useful_bytes`: successfully uploaded missing experts used by the
  next actual demand batch.
- `les_prefetch_useless_bytes`: submitted speculative bytes not useful above,
  including partial failed transfers. Cancelled unstarted work contributes zero.
- `les_prefetch_late_bytes`: useful bytes whose upload future was not ready when
  actual demand arrived; this is a subset of useful bytes, not an extra total.
- `les_prefetch_extra_misses`: actual missing experts that hit in the demand-only
  shadow cache, measured on successful demand completion.
- `les_prefetch_deduplicated`, `les_prefetch_promoted`, `les_prefetch_skipped`:
  matching pending requests, queued-task promotions and policy/admission skips.

Final Lunar Lake OCL Debug validation passed 405/405 focused tests in 32 suites
and the overlapping 182/182 forced grouped/micro-Prefill-fallback subset, with
zero failures/skips. Deterministic gated tests cover priority/promotion,
duplicate versus different requests, cancellation, rollback, full-cache
speculation, pollution attribution and single-upload reuse. The queued-prefetch
epoch recovery check also passed 20 repeated runs. Evidence is in workspace
`les/p5_complete_final_results.xml`, `les/p5_complete_final_fallback_results.xml`
and `les/p5_epoch_final_results.xml`.

OLMoE LLM and Qwen VLM each passed all six modes (off/demand/predict/prefetch/
cache-cold/cache-warm), 18 repeated texts/token counts equal per model, with warm
cache blobs unchanged. The prefetch runs recorded respectively 315/819 early
predictions, 213/702 admitted experts, 676,601,856/1,026,342,912 useful bytes,
19,611,648/120,938,496 useless bytes, 22,880,256/34,320,384 late bytes and 26/34
extra misses; prefetch failures were zero. All 12 mode results passed the P5
early-execution and byte-budget/accounting gates. Evidence is in workspace
`les/results/p5_complete_olmoe_20261102/acceptance.json` and
`les/results/p5_complete_qwen_20261102/acceptance.json`; the existing acceptance
script now offers `--p5` and positive/negative parser self-tests.

P5 is complete within the existing plain-FP16, single-token, OCL qualification
boundary. LES remains default-off. These short sequential Debug samples do not
establish formal performance, model/logit accuracy, complete GPU-suite coverage,
ZE LES support or a new TSan result. P0 gaps remain deferred; larger-Prefill
overlap, cross-layer pools and the full compatibility matrix remain P6-P8.

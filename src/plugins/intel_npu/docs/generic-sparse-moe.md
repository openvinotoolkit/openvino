# Generic batched MoE execution in NPUW

## Scope

NPUW host isolation and device lowering share a model-independent description
of a **supported batched expert graph**. This does not add a public OpenVINO
operation, a universal MoE kernel, exporter adapters, or NPU compiler changes.
It does not support every graph containing experts.

The core contribution provides:

- A shared declarative boundary pattern, independent of architecture-specific
  names or a prescribed softmax/normalization formula.
- Transactional single-token device lowering that selects compressed expert
  weights before decompression and computes the selected expert branches.
- Generic host isolation, grouped/plain weight-expression preservation,
  top-one closure mapping, and token-shape handling for the existing executors.
- Bounded static metadata folding and repeated-partition compatibility checks.

Host expert batching, request caching, and asynchronous prefill already existed.
The [runtime score and cleanup guarantees](moe-runtime-semantics.md) are a
separate prerequisite, not new algorithms introduced by the topology matcher.
Automatic LLM model detection, stage configuration and strategy selection are a
separate integration layer described below; the core passes do not themselves
enable them.

The legacy GPT-OSS, Qwen3 and Gemma4 expert matchers already use structural
patterns. The generic path shares a reduction-rooted pattern for
ScatterElementsUpdate, Transpose, score views, Multiply and ReduceSum between
host isolation, device lowering and eligibility queries. TopK's index output
is matched separately after validating optional integer conversions; the
mixing-score expression is not constrained to the selection formula.
Cross-node shape checks and a bounded traversal of the expert arm retain the
layout, weight-selection and no-escaping-intermediate safeguards. Device graph
construction remains separate and transactional. Legacy router callbacks use a
bounded Scatter-to-reduction lookup and the same boundary recognition.

Grouped weight decompression with an intervening Reshape, plain expert weights,
and sigmoid/bias selection with separately computed mixing scores are concrete
coverage gaps in the legacy implementations, not limitations of the pattern
framework.

### Export and compiler prerequisites

Models exported with the graph contract below can use the generic matcher.
Other exports may need normalization or re-exporting with a compatible exporter;
this runtime change does not convert arbitrary expert-loop or token-dispatch
graphs. Routing-score expressions and shared expert branches remain part of the
input model rather than being reconstructed by an export policy.

Historical validation of the original combined implementation used compatible
artifacts for two trained models and four random architecture fixtures, some
with separate exporter adaptations. That is not six-family out-of-box export
support, and those hardware results have not been rerun on the split commits.

## Semantic contract

For token `t`, the router supplies distinct expert indices `I[t, j]` and mixing
coefficients `S[t, j]`. The output is the sum of
`S[t, j] * expert[I[t, j]](hidden[t])`. Shared/dense experts remain outside this sum.

**Selection logits and mixing scores are different inputs.** Selection may
include an expert bias that must not affect the mixture. The matcher preserves
normalization epsilons, learned scales, and softmax-before/after-TopK semantics.
Scores need not sum to one and may be signed or exactly zero. Non-finite host
router scores are rejected by the runtime prerequisite.

The graph boundary requires:

1. A constant positive K and MAX TopK on the expert axis of `[tokens, E]`.
   Indices must come from output **1**, optionally through i32/i64 conversions.
2. ScatterElementsUpdate v3/v12 on expert axis 1 (or -1), with a zero base;
   v12 must use reduction NONE.
3. Transpose `[1, 0]`, then up to seven data-preserving Reshape/Unsqueeze views to
   `[E, tokens, 1]`, `[E, 1, tokens, 1]`, or `[E, tokens, 1, 1]`.
4. Independent expert-major FFNs, starting with Tile of `[tokens, hidden]` by
   `[E, 1]`. At least two expert MatMuls, supported pointwise activations,
   feature-axis Slice/Split, and singleton-preserving activation views are
   supported. Mixing tokens and features through a reshape is rejected.
5. Multiply by the original scattered scores followed by ReduceSum on expert
   axis 0. Intermediate values may not escape the expert boundary.

Plain and constant-derived expert-major weights are supported. Decompression
chains may contain Convert, Add/Subtract/Multiply and constant-shape Reshape.
Grouped quantization must preserve the leading expert dimension, for example
`[E, out, groups, group_size] -> dequantize -> [E, out, in]`.
Host unrolling reconstructs the complete expression per expert slot, preserving
operand order, dtype, broadcast attributes and shared metadata; it does not
simply strip arithmetic to find an underlying weight Parameter. Shared/tied
expressions are deduplicated and every required closure input is mapped,
including top-one and shared scale/zero-point inputs.

Recognition does **not** guarantee compiler support for every quantization
scheme. Historical NPU artifacts used symmetric INT4, group 128 or channel-wise,
with symmetric INT8 backups where needed.

Expert arithmetic accepts NumPy or no broadcasting, not PDPD explicit-axis
broadcasting. A lower-rank PDPD scale could align with the expert axis and would
need separate selection logic. Arbitrary expert loops, NonZero token dispatch,
cross-token/expert operations, nonzero scatter bases and unrecognized layouts
are conservatively rejected rather than silently rewritten.

## Execution paths

### Device-routed decode

[The topology implementation](../src/plugin/npuw/moe_transformations/moe_topology.cpp)
analyzes without mutation and builds a selected-expert graph. Replacement occurs
only after construction succeeds. Original immutable constants stay shared;
their producers and decompression chains are not modified in place.

Gather selects K slices from packed constants before Convert/decompression,
including expert-specific scales and zero points. The graph uses the original
mixing scores. The existing Gather-to-2D-Gather adaptation widens i32 indices
to i64 before row-offset arithmetic and broadcasts a singleton offset row
without an explicit Tile. The existing integer-constant preservation marker
is retained.

This requires **static single-token decode**, not device-resident ragged
prefill. A requested device pass that lowers no supported block fails. In mixed
graphs, verify the expected number of transformed layers; compilation alone
does not establish full sparsity. `can_device_route()` exposes eligibility to
callers but does not itself choose an LLM strategy.

### Host-routed decode and prefill

The generic `BatchedExpert` matcher tags the isolated expert boundary and router
with K, including when the router is not a folded function. The `MOE` preset
retains the legacy matchers alongside the generic matcher. Regressions compare
expert isolation with the original preset for GPT-OSS, Qwen3 and Gemma4 in
prefill and decode. Expert-tag callbacks are registered once even when multiple
matchers recognize the block. Legacy router callbacks use the explicit
selection TopK rather than a different TopK in the mixing-score expression.

Callers must retain expert-tagged blocks independently of ordinary repeated-block
profitability thresholds when configuring isolation, for example through
`NPUW_ONLINE_KEEP_BLOCKS_TAGGED=expert`. Automatic preservation in LLM stage
configuration is provided by the LLM integration below.

Decode uses the existing K-expert single-inference executable; prefill uses
the existing token-to-expert grouping, chunk requests and output accumulator.
Top-one decode retains an explicit closure mapping. Token resizing follows the
identified activation/router parameters and views, including a converted expert
output. Weights are not classified as token inputs merely because they are
reachable from the mixing operation or have numerically equal dimensions.

Standalone expert graphs use the function-call pipeline and downstream reduction
adaptation. A global hidden-state input is bound through the executor rather
than the untransformed model interface. This path is covered by synthetic native
dispatch tests; it is **not** a claim of a validated trained first-layer workload.

Host routing requires uniform K across matched layers and supported expert
layouts. Packed four-bit expert boundaries must be byte-aligned. An isolated
expert partition with missing K or failed sparse lowering raises an error instead
of silently retaining a dense body. Unrecognized/unisolated graphs are not
thereby guaranteed sparse; inspect actual execution.

## Metadata and repeated partitions

`FoldStaticMoEMetadata` folds static integer/boolean metadata before partitioning
when invoked by the caller. It bounds output size, supports multiple outputs,
and respects disabled folding and `can_constant_fold()`. Stateful operations
such as integer RandomUniform must not be evaluated or frozen. Floating-point
weight decompression is not materialized. Constants stay shared across consumers.

Repeated-function metadata distinguishes constant from runtime integer/boolean
operands. Otherwise constant and runtime attention masks could be grouped into
a function with an inconsistent constant bank. Floating-point closure
compatibility remains unchanged. LLM-stage pass invocation is provided separately.

## LLMCompiledModel integration

The LLM wrapper uses the shared structural pattern to detect compatible batched
MoE blocks while retaining existing legacy router/expert name hints. A name hint
alone does not establish device-lowering eligibility.

After stage shapes are prepared, both `FoldShapeComputeChain` and
`FoldStaticMoEMetadata` run on prefill and every generate variant, before online
partitioning. Host-routed stage configuration appends the `expert` keep tag to
existing tags (including attention tags), rather than replacing them. This keeps
small expert partitions eligible for sparse execution without changing ordinary
repeated-block profitability thresholds. Existing isolation presets and unrelated
user properties remain in the stage configuration.

When no explicit generate MoE hint is supplied, automatic device routing retains
the architecture `5010` and compiler-version-at-least-7.29 gates and additionally
requires a generation-token length of one and eligible supported topology in
**every prepared generate variant**. Otherwise the existing host-routed default
remains. An explicit host hint is not replaced by automatic selection.

Stage configuration is applied after these checks. Device transforms run only
on generate variants. Explicit device prefill and unsupported device decode
requests fail rather than silently selecting a different strategy. Device
prefill remains unsupported, and the existing dense hint remains CPU-only.
For a mixed graph, eligibility is not a claim that every unrecognized block was
lowered: verify transformed-layer counts and placement as described below.

The integration tests cover single-/multi-token generation, architecture/compiler
gates, explicit hints, name-only rejection, unnamed structural detection, multiple
generate variants, small expert blocks, and preservation of other stage settings.
They record prepared models and properties through a test compilation factory;
core tests separately check numerical lowering and actual executor dispatch.

## Validation and diagnostics

Native regressions cover router semantics, distinct expert values, K=1/2/E,
channel-wise/grouped packed weights, shared branches, malformed topology,
token/hidden-size collisions, closure mappings, metadata folding, zero/signed
scores and cache-safe reuse. The sparse-dispatch regression uses evaluatable
submodels through real NPUW/executor paths, including global input and deferred
request cleanup. Some other existing native tests require the CPU plugin;
build the loadable CPU/NPU plugins and IR frontend when running the full suite.

For hardware validation:

- Check the imported runtime and actual device; disable device fallback.
- Use CPU f32 or a recorded GPU reference, teacher-forced logit comparisons,
  free generation with EOS stopping, state reset and transformed-layer counts.
- Require `DeviceRoutedMoE: selected K/E experts` for device decode,
  `MoE Expert Batch` / `Expert Inference` for host decode, and
  `MoE Expert Iterative` / `NPU Start` for sparse prefill.
- Development profiling needs `ENABLE_DEBUG_CAPS` and `ENABLE_NPU_DEBUG_CAPS`.
  A routing hint or fluent output alone does not prove sparse execution.
- Start accuracy checks with `NPUW_DQ=NO` and
  `NPU_COMPILER_DYNAMIC_QUANTIZATION=NO`. These are validation settings, not
  production-default changes; extra activation quantization caused substantial
  errors in the historical pretrained-checkpoint tests.

Random fixtures exercise architecture, not language quality or useful throughput.
Use identical quantization, prompts, reference-token trajectories and properties
for trained-model comparisons. Separate compile time, warm TTFT and decode
latency; sparse prefill can be slower for short prompts despite faster decode.

## Remaining work

- Device-resident token dispatch, grouped/ragged expert prefill and fusion.
- More exporter forms, mixed-K/layout models, platforms and drivers.
- Larger context/workload matrices, cold/warm cache and memory measurements,
  production-quality accuracy/performance tuning, and serialization/cache review.
- Integration with public grouped-MatMul/MoE abstractions as they mature.

Separate exporter/compiler work is outside this runtime contribution.
Significant AI assistance must be disclosed and does not replace contributor
review and validation under the [AI usage policy](../../../../AI_USAGE_POLICY.md).
# Generic batched MoE execution in NPUW

## Scope

This change gives NPUW host and device routing a shared, model-independent
description of a **supported batched expert graph**. It does not add a public
OpenVINO operation, replace the compiler with a universal MoE kernel, or support
every graph containing experts.

The contribution combines:

- A shared declarative boundary pattern, independent of architecture-specific
  names or a prescribed softmax/normalization formula.
- Transactional single-token device lowering that selects compressed expert
  weights before decompression and computes only the selected expert branches.
- Reuse of NPUW's existing sparse host-prefill and K-expert decode executors,
  with grouped/plain weight support, top-one support, and exact zero-score handling.

Host-side expert batching, request caching, and asynchronous prefill already
existed. They are reused here, not claimed as new algorithms.

The legacy GPT-OSS, Qwen3 and Gemma4 expert matchers already use structural
patterns. The generic path shares a reduction-rooted pattern for the
ScatterElementsUpdate, Transpose, score views, Multiply and ReduceSum boundary
between host isolation, device lowering and eligibility queries. TopK's index
output is matched separately after validating optional integer conversions;
the mixing-score expression is not constrained to the selection formula.
Cross-node shape checks and a bounded traversal of the expert arm retain the
layout, weight-selection and no-escaping-intermediate safeguards. Device graph
construction remains separate and transactional. Legacy router callbacks keep
a bounded Scatter-to-reduction lookup, but reuse the same boundary recognition.

Concrete forms beyond the existing legacy skeletons include grouped weight
decompression with an intervening Reshape, plain expert weights, and routers
with sigmoid/bias selection and separately computed mixing scores. These are
coverage gaps in those implementations, not limitations of the pattern framework.

### Export and compiler prerequisites

This is an OpenVINO runtime contribution only. It does not include exporter
adapters, NPU compiler changes, or experimental device-prefill implementations.
The validated runtime paths work with the existing prebuilt NPU compiler.

Models already exported with the graph contract below can use the generic
matcher. Other exports may need normalization or re-exporting with a compatible
exporter; installing this runtime change does not automatically convert arbitrary
expert-loop or token-dispatch graphs. Validation covers two trained models and
four random architecture fixtures using compatible artifacts, some produced with
separate exporter adaptations. This is not six-family out-of-box export support.
Routing-score expressions and shared expert branches remain part of the input
model and are preserved rather than reconstructed by an export policy here.

## Semantic contract

For token `t`, the router provides distinct selected expert indices `I[t, j]`
and mixing coefficients `S[t, j]`. The result is the sum of
`S[t, j] * expert[I[t, j]](hidden[t])` over the selected slots. Shared/dense
experts, if present, remain outside this sum.

**Selection logits and mixing scores are different inputs.** Selection may
include an expert bias that must not affect the mixture. The matcher preserves
the existing score expression, including normalization epsilons, learned
scales, and softmax-before/after-TopK semantics. Scores need not sum to one and
may be signed or exactly zero. Non-finite host router scores are rejected.

The current graph boundary requires:

1. A constant positive K and a MAX TopK on the expert axis of `[tokens, E]`.
   Indices must come from output **1**, optionally through i32/i64 conversions.
2. ScatterElementsUpdate v3/v12 on expert axis 1 (or -1), with a zero base;
   v12 must use reduction NONE.
3. Transpose `[1, 0]`, then up to seven data-preserving Reshape/Unsqueeze views to
   `[E, tokens, 1]`, `[E, 1, tokens, 1]`, or `[E, tokens, 1, 1]`.
4. Independent expert-major FFNs, starting with a Tile of `[tokens, hidden]`
   by `[E, 1]`. At least two expert MatMuls, supported pointwise activations,
   feature-axis Slice/Split, and singleton-preserving activation views are
   supported. Mixing tokens and features through a reshape is rejected.
5. Multiply by the original scattered scores followed by ReduceSum on expert
   axis 0. Intermediate values may not escape the expert boundary.

The matcher supports plain and constant-derived expert-major weights. Supported
decompression chains contain Convert, Add/Subtract/Multiply, and constant-shape
Reshape. Grouped quantization must preserve the leading expert dimension, e.g.
`[E, out, groups, group_size] -> dequantize -> [E, out, in]`. Recognizing a
decompression expression does **not** guarantee the NPU compiler accepts every
quantization scheme; the validated NPU artifacts use symmetric INT4, group 128
or channel-wise, with symmetric INT8 backups where needed.

Expert arithmetic currently accepts NumPy or no broadcasting, not PDPD's
explicit-axis broadcasting. A lower-rank PDPD scale can align with the expert
axis and would need separate selection logic. Such graphs are conservatively
rejected rather than silently reordering scales incorrectly.

Conservative rejection is intentional. Arbitrary expert loops, NonZero-based
token dispatch, cross-token/expert operations, nonzero scatter bases, and
unrecognized layouts require normalization or a separate implementation.

## Execution paths

### Device-routed decode

[The shared topology implementation](../src/plugin/npuw/moe_transformations/moe_topology.cpp)
analyzes the graph without mutation and constructs a new
selected-expert graph. The replacement is installed only after construction
succeeds. Original immutable constants remain shared; their producers and
decompression chains are not modified in place.

Gather selects K expert slices from packed constants before Convert and
decompression, including expert-specific scales/zero points. The new graph
uses the original selected scores rather than reconstructing a router formula.
The existing Gather-to-2D-Gather pass adapts the graph to NPU compilation.
It widens i32 routing indices to i64 before computing expanded row offsets.

This path requires **static single-token decode**. It is not device-resident
ragged prefill. A requested device pass that lowers no supported expert block
fails explicitly. In a mixed graph, verify the expected number of transformed
layers; a successful compilation alone does not establish full sparsity.
Automatic strategy selection checks the prepared static graph and avoids device
routing for multi-token generation. Explicit unsupported device requests still
produce a clear error.

### Host-routed decode and prefill

The generic `BatchedExpert` matcher tags the isolated expert boundary and router
with K. Tagging the boundary is necessary even if the router itself is not a
folded function. The expert tag is preserved independently of generic
repeated-block profitability thresholds. Legacy named matchers remain available
in the default `MOE` preset alongside the generic matcher, as well as explicitly.
Regression tests compare expert isolation with the original preset for GPT-OSS,
Qwen3 and Gemma4 in prefill and decode. Expert-tag compilation callbacks are
registered once, even when several matchers recognize the same expert block.
For a generically recognized block, legacy router callbacks also use the
explicit expert-selection TopK, not a different TopK in the score expression.

Decode uses the existing single inference call for K experts. Only exactly zero
scores are skipped. If fewer than K nonzero contributions remain, unused
compiled slots bind an already selected expert and a zero score. Cache keys use
the padded weight selection; scores and input/output tensors are rebound on
every invocation. An all-zero mixture clears the output without expert
inference. Top-one decode retains an explicit closure mapping.

Prefill uses the existing token-to-expert grouping, chunk executables,
asynchronous requests, and output accumulator. All-zero mixtures return a
cleared accumulator. Standalone expert graphs use the full function-call
pipeline, including downstream reduction adaptation, just as LLM execution does.
Global expert inputs are bound through the executor rather than the ordinary
untransformed model interface.
Token resizing uses the identified activation/router parameters, including when
the expert output has a precision Convert. Weights are never classified as token
inputs merely because they are reachable from the mixing operation.
On exceptional prefill exit, outstanding chunk requests are drained before
their tensors and requests can be reused; partial outputs are not scattered.

Host routing currently requires a uniform K across matched layers and the
supported expert layouts. Packed 4-bit expert boundaries must be byte-aligned.
An isolated expert partition with missing K or failed sparse lowering raises an
error instead of silently using the dense expert body. Unrecognized/unisolated
graphs are not thereby guaranteed sparse; inspect execution evidence.

## Metadata and repeated partitions

Static integer/boolean metadata is folded before partitioning, with a bounded
output size, support for multiple outputs, and respect for both disabled folding
and an operation's `can_constant_fold()` contract. Stateful operations such as
integer RandomUniform must not be evaluated or frozen by this pass.
Floating-point weight decompression is not materialized. Folded constants stay
shared across consumers; per-consumer graph cloning is not needed by the
validated model matrix once metadata compatibility is handled below.

Repeated-function metadata distinguishes constant from runtime **integer and
boolean** operands. Otherwise a constant attention mask and a runtime mask can
be grouped into a function with an inconsistent constant bank. Floating-point
closure compatibility remains unchanged.

## Validation and diagnostics

Tests cover different router semantics, distinct expert values, K=1/2/E,
channel-wise/grouped packed weights, shared branches, malformed topology,
token/hidden-size collisions, closure mappings, metadata folding, and cache-safe
zero/signed scores. The source NPU suite should be run with the CPU plugin built,
because some existing tests explicitly require CPU as a reference device.
The native sparse-dispatch regression itself uses evaluatable test submodels:
it verifies actual sparse dispatch counts, changing cached weights/scores,
global-input binding and deferred-request cleanup without an NPU or CPU plugin.

For hardware validation:

- Check the imported runtime and actual execution device. Disable device fallback.
- Use CPU f32 or a recorded GPU reference, teacher-forced logit comparisons,
  and a separate free-generation check. Stop chat generation at EOS.
- Verify state reset and the expected transformed-layer count.
- Require `DeviceRoutedMoE: selected K/E experts` for device decode, or
  `MoE Expert Batch` / `Expert Inference` for host decode; require
  `MoE Expert Iterative` / `NPU Start` for sparse prefill.
- Both `ENABLE_DEBUG_CAPS` and `ENABLE_NPU_DEBUG_CAPS` are needed for the
  development profiling used by these checks. A routing hint or correct text
  alone is not evidence that the sparse executor ran.
- Start accuracy checks with `NPUW_DQ=NO` and
  `NPU_COMPILER_DYNAMIC_QUANTIZATION=NO`. These are validation settings, not
  a change to production defaults. Additional activation quantization caused
  substantial errors in the tested pretrained checkpoints.

Tiny random models exercise architectures, not language quality or useful
throughput. Benchmark trained models using identical quantization, prompt,
reference-token trajectory, and runtime properties. Separate compile time,
warm TTFT, and decode latency. Sparse prefill can be slower than dense prefill
for short prompts even when sparse decode is substantially faster.

## Remaining work

- Device-resident token dispatch, grouped/ragged expert prefill, and fusion.
- Additional exporter forms, mixed K/layout models, and platforms/drivers.
- Larger prompt/context and workload matrices, cold/warm cache and memory
  measurements, and production-quality accuracy/performance tuning.
- Review of serialization/cache compatibility and integration with public
  grouped-MatMul/MoE abstractions as those interfaces mature.

Before upstream submission, keep the runtime prerequisites, topology and executor
changes, and their regressions in reviewable commits. Separate exporter or
compiler work is outside this contribution. Significant AI assistance must be
disclosed, and the contributor must perform human review and validation under
OpenVINO's [AI usage policy](../../../../AI_USAGE_POLICY.md). No benchmark here
substitutes for that review.
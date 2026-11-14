# MOC transformation performance

## Scope and conclusion

This audit covers every pass registered by `MOCTransformations` and
`MOCLegacyTransformations`, including their nested pipelines, at upstream
commit `7d6e434920f20f23f33ab2077e29a8611f7f7697`.
The objective is transformation latency, not inference throughput.

The first optimization should be **better matcher dispatch, not a wholesale
conversion of ModelPasses to MatcherPasses**. Most local MOC rewrites already
are MatcherPasses, and many already share GraphRewrite traversals. A standalone
MatcherPass still gets its own GraphRewrite traversal when run by Manager.

GraphRewrite previously used type-based dispatch only when *every* enabled
matcher had an operation or WrapType root. One Or root disabled that fast path
for the entire container. This affected CommonFusions, CommonEliminations,
the decomposition group, and nested NopElimination runs.

The accompanying change recognizes Or roots recursively when all alternatives
are type-based. It keeps existing traversal order, matcher registration order,
first-success behavior, subgraph recursion, and registered-new-node processing.
Untyped alternatives, Optional roots, and custom handlers without patterns
retain the original fallback. Overlapping alternatives dispatch a matcher only
once per node, including when both a base and derived type are eligible.

No MOC pass is reordered, removed, or given a different public base class.

## Cost model and barriers

For N nodes and M matchers, fallback dispatch makes up to N*M matcher attempts;
type-based dispatch visits the same graph but attempts only eligible matchers.
It still pays for queue construction, RTTI ancestor lookup, pattern matching,
and any actual rewrites. Graph traversal and matching are not the only costs:
constant evaluation, tensor copying, shape inference, and symbolic propagation
can dominate models with large weights or complex shape computations.

Important boundaries that a larger reorganization must preserve:

* TransposeConvert and TransposeMatMul run before the first ConstantFolding to
  avoid materializing transposed MatMul weights.
* Dequantization and ShapeOf/RandomUniform folding restrictions precede folding.
* Zero-dimension and subgraph-interface cleanup precede the first folding;
  duplicate TI inputs must be removed before dangling parameters/results.
* Filtering-box fusion precedes simplification that can destroy its pattern.
* Recurrent cell and sequence fusions precede no-op elimination.
* TransposeSinking and SplitSqueezeConcatFusion intentionally share a traversal.
* Shape simplification has an explicit validation boundary before
  GatherNopElimination.
* Arithmetic decomposition precedes linear/convolution multiplication fusion;
  folding then exposes patterns for FakeQuantize fusions.
* SDPA requires symbolic propagation; reverse-input-channel fusion has
  distinct initialization, forward, backward, and finalization phases.
* `use_shapes=false` temporarily relaxes parameter/variable dimensions, retains
  dimension symbols, then restores original shapes and validates.

Per-pass validation is already disabled in MOC, shape simplification, and
SymbolicOptimizations. Disabling it again does not remove another traversal.
Deleting explicit validation or folding boundaries is not justified by this audit.

## Complete top-level inventory

M = MatcherPass; G = GraphRewrite; P = ModelPass. A G container's children are
flattened into a parent GraphRewrite when registered with `add_matcher`.
Rows containing several names describe the same recommendation for each pass,
not a proposal to combine them automatically.

| Stage | Passes and current implementation | Performance assessment |
|---|---|---|
| Initialization | InitNodeInfo (P) | Initializes every node's provenance, recursively. Keep model-wide markup before rewriting. |
| Pre-fold transpose handling | TransposeConvert, TransposeMatMul (M) | Two traversals. Candidate for a dedicated G after checking shared inputs and created-node visibility; keep before folding. |
| Low-precision markup | MarkGatherSubgraph (M); MarkDequantization (M, two configurations) | Preserve configuration order, precision sets, and folding restrictions. Consider shared traversal only after markup-interaction tests. |
| Shape-folding control | DisableShapeOfConstantFolding (M, two conditional registrations) | The second run covers nodes introduced by shape simplification; do not simply remove it. |
| Early elimination | EliminateConvert, EliminateScatterUpdate, RemoveConcatZeroDimInput, EliminateLoopInputsOutputs, EliminateDuplicateTIInputs (M) | Potential batching, but body-descriptor and empty-tensor dependencies prevent treating the whole sequence as independent. |
| Interface cleanup | RemoveMultiSubGraphOpDanglingParamsResults (P) | Reverse-order traversal and body input/output descriptor changes. Keep P. |
| Empty/constant bodies | FoldSubgraphEmptyInputs (M), DisableRandomUniformConstantFolding (M), PushConstantToSubgraph (P) | Keep before folding. P edits body parameters and outer descriptors; changing its base does not make the work local. |
| First folding | ConstantFolding (P) | Global evaluation/bounds work, not an ordinary local matcher. Preserve its boundary. |
| Filtering boxes | FuseFilteringBoxesBySize (G) | Complex early pattern and dynamism handling; keep its position. |
| Smart reshape | LSTMStatesBroadcast (P), ReshapeSinkingMatMul (M), conditional on `!use_shapes` | P tracks batch symbols through TI bodies with temporary shapes and validation. Not a simple matcher conversion. |
| Quantization canonicalization | ConvertQuantizeDequantize (M) | Keep before shape simplification and common fusions. |
| Shape graph | SimplifyShapeOfSubGraph (P) | Orchestrator containing matching, global deduplication, and validation; optimize children, not its public base. |
| NMS indices | ConvertNmsGatherPathToUnsigned (G) | Already shares initialization/propagation/conversion matchers. Markup has forward dependencies. |
| Slices | StridedSliceOptimization (P) | Contains local elimination plus global sibling grouping/deduplication. Retain its staged orchestration. |
| Broadcasting | BroadcastElementwiseFusion, BroadcastMatMulFusion, BroadcastTransition (M) | Or-root dispatch helps the latter two. Batching may alter which broadcast edges are removed or moved; requires interaction tests. |
| Reductions | PullThroughReduce (G) | Already groups Unsqueeze/Reshape-through-reduction matchers. |
| Recurrent fusions | LSTMCellFusion (G), GRUCellFusion (M), SequenceFusion (M) | Keep before no-op elimination. Possible batching needs cell-to-sequence and generated-node tests. |
| Concat simplification | ConcatToBroadcast (M) | Local rewrite; retain position before transpose cleanup. |
| Transpose cleanup | TransposeSinking (G), SplitSqueezeConcatFusion (M), in one G | Existing sharing is intentional. Preserve created-node registration. |
| MatMul cleanup | TransposeMatMul (M, second registration) | Cleans up transposes introduced since the first invocation; not redundant. |
| CommonEliminations | EliminateUnsqueezeGather (M), NopElimination (G), SelectWithOneValueCondition (M) | Already one traversal. Or-root dispatch also accelerates nested NopElimination uses elsewhere. |
| Second folding | ConstantFolding (P) | Preserve boundary before common fusions. |
| CommonFusions: conversions and activations | ConvertScatterElementsToScatter, SoftPlusFusion, SoftPlusToMishFusion (M); SwishFusion, HSwishFusion, HSigmoidFusion (G); NormalizeL2Fusion, ClampFusion (M) | Already share a traversal. NormalizeL2 and Clamp have typed Or alternatives; they should not force all sibling matchers into fallback. |
| CommonFusions: spatial/normalization | PadFusion (G), SoftmaxFusion, ReduceReshapeFusion (M), MVNFusion (G), DilatedConvolutionConverter (M) | MVNFusionWithoutConstants has Divide/Multiply alternatives. Typed dispatch avoids testing unrelated nodes. |
| CommonFusions: activations/control flow | GeluFusion (G), LeakyReluFusion, RandomUniformFusion (M), ConvertTensorIteratorToSequence (G) | GELU has repeated Multiply alternatives. Deduplicate dispatch indices without changing alternative matching order. |
| CommonFusions: interpolation/convolution | SplitConcatPairToInterpolateFusion, ConvolutionToGroupConvolutionFusion, NearestNeighborUpsamplingFusion (M; last conditional on `use_shapes`) | Keep conditional shape behavior and existing shared traversal. |
| CommonFusions: arithmetic/layout | DivideFusion, SubtractFusion, TransposeToReshape, ReshapeSequenceFusion, MatMulConstTransposesExtraction (M) | Already batched. Do not confuse these with later arithmetic decomposition passes. |
| CommonFusions: remaining local patterns | PReluFusion (G); DepthToSpaceFusion, ShuffleChannelsFusion, NonZeroHorizontalFusion, AdaptivePoolToReduce, ConvertU4WeightsZeroPointToScalar (M) | Keep current sharing. NonZero horizontal fusion inspects siblings, so batching new peers would need consumer-interaction analysis. |
| Attention packing | PackMultiHeadAttention (P) | Six successive matching passes with generated-pattern dependencies and manager validation. Candidate for phase-level work, not automatic flattening. |
| Attention fusion | SDPAFusion (P) | Symbolic context plus four matching stages. Keep propagation before shape-symbol predicates. |
| Binary convolution | BinarizeWeights, ConvToBinaryConv (M) | Potential G, but weight canonicalization/created nodes must remain visible to the second pass. |
| Decomposition | BatchNormDecomposition, ConvertDivideWithConstant, ConvertSubtractWithConstant, ConvertNegative, ConvertConvertPromoteTypes (M, one G) | BatchNormDecomposition's Or of opset versions previously disabled typed dispatch for the group. |
| Linear arithmetic | LinOpSequenceFusion (G) | Already shares multiplication/addition simplifications. Preserve after decomposition. |
| MultiplyFusions | ConvolutionMultiplyFusion, GroupConvolutionMultiplyFusion, ConvolutionBackpropDataMultiplyFusion, GroupConvolutionBackpropDataMultiplyFusion, MultiplyConvolutionFusion, MultiplyGroupConvolutionFusion, MultiplyConvolutionBackpropDataFusion, MultiplyGroupConvolutionBackpropDataFusion (M, one G) | Already one traversal. Some patterns change earlier convolution nodes; keep their relative order and shape checks. |
| Third folding | ConstantFolding (P) | Folds arithmetic/weight changes before quantization fusions. |
| FakeQuantizeFusions | FakeQuantizeConcatFusion, FakeQuantizeMulFusion, FakeQuantizeReshapeFusion, PullTransposeThroughFQUp, ReluFakeQuantizeFusion, FuseClampAndFakeQuantize, AddFakeQuantizeFusion, MulFakeQuantizeFusion (M, one G) | Already batched. Keep before rank alignment/deduplication/final folding. |
| Input channels | ReverseInputChannelsFusion (P) | Global multi-direction attribute propagation. Its generic handlers intentionally require fallback. Do not flatten its forward/backward phases. |
| Rank alignment | AlignEltwiseInputRanks (M) | Local constant reshaping; currently reports false even after modifications. Audit change reporting before attempting new batching. |
| Deduplication | SharedOpOptimization (P) | ShapeOf upgrading plus sibling comparison using topological indices and attribute caches. Keep global semantics. |
| Fourth folding | ConstantFolding (P) | Final explicit MOC folding boundary; nested pipelines also have their own evaluation work. |
| Symbolic optimization | SymbolicOptimizations (P) | Propagation/equivalence/value optimization and another shape-simplification pipeline. Major remaining traversal cost, but stages depend on global symbolic state. |
| Naming | ResolveNameCollisions (P) | Global uniqueness maps across model and bodies. Not a local rewrite. |
| MoE | FuseMOE (P), VectorizedMOE2GEMMTransposeWeights (M) | First is an orchestrator for FuseMOEExperts. Keep its matching before weight transposition; folding large expert tensors can dominate. |
| Validation/restoration | Validate (P), final `validate_nodes_and_infer_types()` | Preserve shape restoration and validation; measure before considering removal of potentially overlapping work. |
| Legacy entry point | MOCLegacyTransformations (P) -> ChangePlaceholderTypes (P) | Iterates parameters/consumers rather than the whole graph. Wrapping it in G would add unnecessary node visits. |

### Nested pipelines

* **SimplifyShapeOfSubGraph:** PrepareShapeOpsForEliminationAroundBE, AbsSinking,
  SharedOpOptimization, EliminateGatherUnsqueeze, NopElimination,
  GroupedGatherElimination, Validate, GatherNopElimination,
  SimplifyGatherShapeOf, SimplifySecondInputOfReshape, Validate.
  Keep deduplication before gather/unsqueeze elimination and validation before
  shape-dependent gather elimination. Contiguous local stages are follow-up
  batching candidates, not proven independent stages.
* **StridedSliceOptimization:** with shapes, UselessSliceEraser,
  SharedOpOptimization, GroupedStridedSliceOptimizer,
  GroupedSliceToVSplitOptimization; then SliceSequenceToSingleSlice.
  UselessSliceEraser is a plausible P-to-M conversion because its action is
  local, but grouping slice siblings still requires global state. Match the
  original recursion policy and Result-facing name handling if converting it.
* **PackMultiHeadAttention:** MergeUnrolledSDPA, MergeKVCaches,
  MergeUnrolledRoPE, MergeLinearProjections, MergeDQ, ConcatFusion.
  These merge related heads/projections in sequence; one node-wise traversal
  is not equivalent to six full passes.
* **SDPAFusion:** SymbolicOptimizations(false) supplies SymbolicPropagation and
  ApplySymbolEquivalence, followed by SDPAFusionMatcher,
  SDPASplitAttentionFusionMatcher, SDPAFusionMatcherSinks, SDPAReshapeFusion.
* **ReverseInputChannelsFusion:** SplitConcat/Gather initialization; forward
  Convolution, GroupConvolution, Binary, ShapeOf, Transpose, PassThrough,
  Unsupported handlers; backward Binary/ConvertPassThrough plus Constant
  handling; InsertReverseInputChannel, EraseSplitConcat, EraseGather finalization.
  Early exit after unsuccessful initialization is already present.
* **SymbolicOptimizations:** SymbolicPropagation, ChainedMaximumOptimization,
  NopBroadcast, NopElimination, SharedOpOptimization, ApplySymbolEquivalence,
  OptimizeSymbolsUsedAsValues, LabelResolvingThroughSelect, DeReshapeMatMul,
  DeReshapeFullyConnected, ReshapeOptimizations, SimplifyShapeOfSubGraph.
  The full-run-only stages must not be pulled into the partial SDPA context.
* Other G containers above already flatten their local pattern variants:
  recurrent-cell, activation, normalization, padding, no-op, reduction, and
  TI-to-sequence variants. Splitting them into individual Manager registrations
  would increase traversals rather than improve performance.

## Follow-up priorities

1. Retain type-based dispatch for fully typed Or alternatives (implemented).
2. Profile the remaining symbolic/shape passes with real pre-MOC graphs; measure
   time in inference and evaluation separately from matching.
3. Experiment with small, contiguous batches, starting with pre-fold transpose
   handling and binary convolution. Compare optimized graphs, tensor/runtime
   names, disabled-pass configurations, and nested bodies before accepting them.
4. Consider converting UselessSliceEraser, but only together with a useful shared
   traversal. Changing P to M in isolation still performs a full traversal.
5. Defer global-state/descriptor/forward-backward passes and folding/validation
   boundary removal. A different base class does not eliminate their work.

Changing the exported base class of an existing transformation can also affect
ABI and external pass registration. Avoid that merely to obtain a different
internal traversal.

## Measurement

Use a Release build and the same compiler, libraries, CPU affinity, and workloads
for both revisions. Load or construct the input model once; clone it **outside**
the timed interval for every iteration. Time Manager construction plus
MOCTransformations and its final validation, with three warmups and at least
21 measured iterations. Test both `use_shapes` modes with low precision enabled.
Do not time file loading, graph construction, or serialization as MOC.

For stage-level diagnostics use `OV_ENABLE_PROFILE_PASS=/absolute/profile.csv`.
The semicolon-delimited `t` records contain nanoseconds, pass name, and manager
name. Nested managers overlap their parent timings; do not sum both as total
pipeline time. Repeated passes such as ConstantFolding share a name.

### Results

Release, GCC 11, Linux x86-64, Intel Core i9-12900K; single process pinned to CPU 2.
Numbers are median milliseconds over 21 measured iterations after three warmups.
Baseline/optimized execution order alternates by workload. Low precision remains
enabled. Graph cloning and serialization are excluded.

| Workload (input nodes) | use_shapes | Baseline ms | Optimized ms | Latency reduction |
|---|---|---:|---:|---:|
| 1,000 Tanh chain (1,002) | true | 37.576 | 16.122 | 57.1% |
| 1,000 Tanh chain (1,002) | false | 34.084 | 15.705 | 53.9% |
| 5,000 Tanh chain (5,002) | true | 191.168 | 82.767 | 56.7% |
| 5,000 Tanh chain (5,002) | false | 169.611 | 77.344 | 54.4% |
| MobileNetV2 INT8 IR (1,258) | true | 42.456 | 24.574 | 42.1% |
| MobileNetV2 INT8 IR (1,258) | false | 40.254 | 24.986 | 37.9% |
| SDPA pre-fusion IR (1,561) | true | 77.785 | 45.929 | 41.0% |
| SDPA pre-fusion IR (1,561) | false | 72.236 | 44.717 | 38.1% |

Serialized XML and BIN outputs are byte-identical between revisions for all
four workloads in both shape modes. Output node counts are unchanged:
1,002 / 5,002 / 844 / 1,513 respectively. The core suite passes 14,170 tests
(eight skipped); the transformation suite passes all 10,352 executed tests.

A separate, unpinned profiling run on the 1,000-Tanh graph attributes the gain
to matcher-heavy stages: CommonFusions averages 8.83 -> 0.10 ms,
CommonEliminations 2.82 -> 0.06 ms, SimplifyShapeOfSubGraph 3.35 -> 0.67 ms,
and SymbolicOptimizations 10.13 -> 3.40 ms. These are diagnostic averages
across both shape modes, not additional measurements to sum with total time.

These are local transformation-latency results, **not inference speedups**.
Synthetic chains isolate overhead and contain no useful fusion opportunities.
The two existing IRs exercise actual rewrites but may already be partly
optimized; a broader set of freshly converted pre-MOC models is still needed
before extrapolating to arbitrary models.

### Reproducing the synthetic measurement

Build `openvino` in Release. Compile this standalone harness against that build,
using the core, inference, and transformations include directories and linking
`libopenvino`. Set the runtime library search path to the build being measured.
Run `taskset -c 2 ./moc_benchmark 1000` and repeat with `5000`.

```cpp
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/moc_transformations.hpp"

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    auto input = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32, ov::Shape{1, 16});
    std::shared_ptr<ov::Node> output = input;
    for (size_t i = 0; i < std::stoul(argv[1]); ++i)
        output = std::make_shared<ov::op::v0::Tanh>(output);
    auto original = std::make_shared<ov::Model>(
        ov::OutputVector{output}, ov::ParameterVector{input});
    for (bool use_shapes : {true, false}) {
        std::vector<double> times;
        for (size_t i = 0; i < 24; ++i) {
            auto model = original->clone();
            auto start = std::chrono::steady_clock::now();
            ov::pass::Manager manager;
            manager.register_pass<ov::pass::MOCTransformations>(use_shapes);
            manager.run_passes(model);
            auto end = std::chrono::steady_clock::now();
            if (i >= 3)
                times.push_back(std::chrono::duration<double, std::milli>(
                    end - start).count());
        }
        std::sort(times.begin(), times.end());
        std::cout << use_shapes << ": " << times[times.size() / 2] << " ms\n";
    }
}
```

For IR measurements replace graph construction with `ov::Core::read_model`
before the timing loop. Use the same input files on both revisions.

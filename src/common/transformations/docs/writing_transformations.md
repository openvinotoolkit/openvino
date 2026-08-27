# Writing Transformations for OpenVINO

## Contents

- [Where transformations live](#where-transformations-live)
- [Quick checklist](#quick-checklist)
- [Choosing the pass type](#choosing-the-pass-type)
- [Anatomy of a MatcherPass](#anatomy-of-a-matcherpass)
- [Pattern matching](#pattern-matching)
  - [Predicate reference](#predicate-reference)
  - [Optional nodes and alternatives](#optional-nodes-and-alternatives)
  - [Shape symbols](#shape-symbols)
  - [Pattern blocks](#pattern-blocks)
- [Writing the callback](#writing-the-callback)
- [Modifying the graph](#modifying-the-graph)
- [Low precision specifics](#low-precision-specifics)
- [Internal operations (`ov_ops`)](#internal-operations-ov_ops)
- [Registering the pass in a pipeline](#registering-the-pass-in-a-pipeline)
- [Performance considerations](#performance-considerations)
- [Documenting the pass](#documenting-the-pass)
- [Naming](#naming)
- [Testing](#testing)
- [Reusable utilities](#reusable-utilities)
- [What NOT to do](#what-not-to-do)

## Where transformations live

| Kind | Location |
|------|----------|
| Device-agnostic optimizations | [src/common/transformations/src/transformations/common_optimizations/](../src/transformations/common_optimizations/) |
| Opset conversions / decompositions | [src/common/transformations/src/transformations/op_conversions/](../src/transformations/op_conversions/) |
| FP16/BF16 compression markup | [src/common/transformations/src/transformations/fp16_compression/](../src/transformations/fp16_compression/) |
| Low precision (LPT) | [src/common/low_precision_transformations/](../../low_precision_transformations/) |
| Internal operations | [src/common/transformations/include/ov_ops/](../include/ov_ops/) |
| Reusable pattern blocks | [src/common/transformations/include/transformations/pattern_blocks/](../include/transformations/pattern_blocks/) |
| Plugin-specific passes | `src/plugins/<plugin>/src/transformations/` |

A pass that is beneficial regardless of the backend belongs in common transformations, **not** duplicated per plugin. Express device-specific behavior as a pass parameter (e.g. a list of supported precisions) instead of forking the pass.

Build and run the corresponding tests with:

```bash
cmake --build build --target ov_transformations_tests -j$(nproc)
./bin/ov_transformations_tests --gtest_filter="MyTransformation*"
```

## Quick checklist

1. Pick the right base class: `MatcherPass` for local rewrites, `ModelPass` when nodes outside the matched pattern are modified
2. Put **every** applicability condition into the pattern, not into the callback
3. Prefer built-in predicates (`shape_matches`, `type_matches_any`, `attrs_match`, …) over custom lambdas
4. Access mandatory pattern nodes with `pattern_map.at(label)`; assert on invariants instead of `return false`
5. Call `copy_runtime_info` for every created node and preserve friendly names
6. Solve the general problem — no topology-specific proxies, no hardcoded model-specific values
7. Justify every restriction (consumer count, per-tensor only, const-only, static-shape-only)
8. Register the pass in **all** pipelines that need it
9. Describe what the pass does and why in the header, with a subgraph scheme
10. Add tests per [Writing transformation tests](./writing_tests.md)

## Choosing the pass type

| Pass type | Use when |
|-----------|----------|
| `ov::pass::MatcherPass` | The rewrite touches only nodes contained in the matched pattern. This is the default choice. |
| `ov::pass::ModelPass` | The rewrite modifies nodes outside the pattern (consumers of the match root, distant producers, model inputs/outputs/sinks), or needs global model state. |
| `ov::pass::GraphRewrite` | Several independent matchers should run in one graph traversal. Prefer this over branching on variants inside a single callback. |
| `ov::pass::BackwardGraphRewrite` | The matchers require bottom-up traversal. |

Two rules follow from this table:

- **A pass has a single responsibility.** Long `if (is_4d) … else if (is_5d) …` chains inside one callback signal that the pass is doing two jobs — split them into separate matchers registered in one `GraphRewrite`, sharing a common helper.
- **A pass that requires a specific traversal order should encapsulate that requirement**, so call sites cannot register it incorrectly:

```cpp
class ov::pass::RMSFusion : public ov::pass::BackwardGraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("RMSFusion");
    RMSFusion() {
        add_matcher<RMSFusionMatcher>();
    }

private:
    class RMSFusionMatcher;   // implementation detail
};
```

Prefer composing existing generic passes over re-implementing their effect inline. For example, a conversion pass should emit the straightforward `Transpose -> MatMul -> Transpose` form and let `TransposeToReshape`, `TransposeFusion` and `ReshapeFusion` clean it up, instead of open-coding transpose elimination.

## Anatomy of a MatcherPass

Header — declaration, visibility macro, RTTI and documentation:

```cpp
// src/common/transformations/include/transformations/common_optimizations/my_fusion.hpp
#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
class TRANSFORMATIONS_API MyFusion;
}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief One-line summary of what the pass does.
 *
 * Why it is needed, followed by a before/after scheme (see "Documenting the pass").
 */
class ov::pass::MyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MyFusion");
    MyFusion();
};
```

Source — `MATCHER_SCOPE`, pattern, callback, registration:

```cpp
// src/common/transformations/src/transformations/common_optimizations/my_fusion.cpp
#include "transformations/common_optimizations/my_fusion.hpp"

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern;

ov::pass::MyFusion::MyFusion() {
    MATCHER_SCOPE(MyFusion);   // defines `matcher_name`, enables conditional compilation

    auto weights_m = any_input(type_matches(element::i8) && shape_matches("OC, IC, 3, 3"));
    auto conv_m    = wrap_type<ov::op::v1::Convolution>({any_input(), weights_m},
                                                        {{"strides", Strides{1, 1}}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();
        // ... build the replacement
        return true;
    };

    auto m = std::make_shared<Matcher>(conv_m, matcher_name);
    register_matcher(m, callback);
}
```

`MATCHER_SCOPE` / `RUN_ON_MODEL_SCOPE` are mandatory: they provide the ITT domain name used by transformation profiling and allow the pass to be excluded from conditionally-compiled builds.

Use `OV_CAPTURE_CPY_AND_THIS` in the callback lambda capture list instead of `[=]` or `[&]`.

## Pattern matching

**Every condition that decides whether the pass applies must be expressed in the pattern** — operation type, rank, shape, element type, attribute values, constness, staticness. The callback is responsible only for *building the replacement*.

Conditions in the pattern are self-documenting, composable, enforced by the matcher (so the callback needs no defensive code), and — crucially — visible in [matcher logs](./debug_capabilities/matcher_logging.md), which turns "why didn't my pass fire?" into a log-reading exercise instead of a debugging session.

Avoid:

```cpp
auto conv_m = wrap_type<v1::Convolution>();

ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
    auto conv = ov::as_type_ptr<v1::Convolution>(m.get_match_root());
    if (!conv)
        return false;
    if (conv->get_strides() != Strides{1, 1})
        return false;
    const auto& w = conv->get_input_partial_shape(1);
    if (w.rank().is_dynamic() || w.rank().get_length() != 4 || w[2] != 3 || w[3] != 3)
        return false;
    if (conv->get_input_element_type(1) != element::i8)
        return false;
    // ... actual rewrite
};
```

Prefer:

```cpp
auto weights_m = any_input(type_matches(element::i8) && shape_matches("OC, IC, 3, 3"));
auto conv_m    = wrap_type<v1::Convolution>({any_input(), weights_m}, {{"strides", Strides{1, 1}}});

ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
    const auto& pattern_map = m.get_pattern_value_map();
    const auto& symbols     = m.get_symbols();   // "OC" / "IC" available for further checks
    // ... actual rewrite only
};
```

The same rule applies to plugin pipelines: manual model traversal used to decide whether to register a pass should instead become a pass parameter plus a pattern predicate.

### Predicate reference

Declared in [openvino/pass/pattern/op/pattern.hpp](../../../core/include/openvino/pass/pattern/op/pattern.hpp). Combine them with `&&`, `||` and `!`.

| Need | Predicate |
|------|-----------|
| Shape / dimension relations, symbol capture | `shape_matches("B, ?, 1, 1")`, `shape_matches("Batches..., M, N")` |
| Constant value relations | `value_matches("[1, 2]")` |
| Rank only | `rank_equals(3)`, `rank_more_than(1)` |
| Element type | `type_matches(element::i8)`, `type_matches_any({element::u8, element::i8})` |
| Node attributes | `attrs_match({{"mode", "numpy"}})`, or the `attrs` argument of `wrap_type` |
| Static shape / rank | `has_static_shape()`, `has_static_rank()`, `has_static_dim(pos)`, `has_static_dims({...})` |
| Consumers | `consumers_count(1)`, `consumers_more_than(1)` |
| Specific output port | `output_index_matches(0)` |

Node builders:

| Need | Builder |
|------|---------|
| Node of one of several types | `wrap_type<v0::Sin, v0::Cos>({input})` |
| Any node with a predicate | `any_input(pred)` |
| Any `Constant` | `wrap_const()` |
| Optional node in the chain | `optional<v1::Subtract>({input, any_input()})` |
| Alternatives | `a \| b` |

Custom `pattern::Predicate` lambdas should be a last resort. The built-ins already handle dynamic dimensions, symbol propagation and precision corner cases that hand-written checks tend to get wrong.

### Optional nodes and alternatives

Do not hand-roll `Or` chains or duplicate a matcher for a "with/without X" variant:

```cpp
// Avoid
auto with_sub    = wrap_type<v1::Subtract>({convert, zp});
auto without_sub = convert;
auto sub_or_not  = std::make_shared<pattern::op::Or>(OutputVector{with_sub, without_sub});

// Prefer
auto sub_m = optional<v1::Subtract>({convert, zp});

// Alternatives use operator|
auto weights_m = weights_5d_convert_m | weights_4d_convert_m;
```

Presence of an optional node is checked in the callback with `pattern_map.count(sub_m)`.

### Shape symbols

`shape_matches` captures named dimensions that become available through `Matcher::get_symbols()`. Use them instead of index arithmetic on shapes, and give them domain names (`hidden_in`, `seq_len`, `OC`, `IC`) rather than `?`:

```cpp
auto weights_m = any_input(has_static_shape() && shape_matches("OC, IC, 3, 3"));
auto conv_m    = wrap_type<v1::Convolution>({any_input(), weights_m}, {{"strides", Strides{1, 1}}});

ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
    const auto& symbols = m.get_symbols();
    if (symbols.at("OC").i() < 512 || symbols.at("IC").i() < 512)
        return false;
    ...
};
```

Ellipsis notation (`"Batches..., M, N"`) matches a variable number of leading dimensions and lets the same name be reused across several pattern inputs to require equality.

For dynamic shapes, `ov::symbol::util::dims_are_equal` compares dimensions that are equal by symbol — this often removes the need for a static-shape restriction.

### Pattern blocks

Recurring subgraph shapes (compressed weights, dequantization chains, MLP blocks) are described once as a `pattern::op::Block` and reused. See [pattern_blocks/](../include/transformations/pattern_blocks/) — e.g. `CompressedWeightsBlock`. Add a new block instead of re-describing the same subgraph in a third pass.

## Writing the callback

### Access matched nodes through the pattern map

```cpp
const auto& pattern_map = m.get_pattern_value_map();
const auto weights = pattern_map.at(weights_m);            // mandatory node -> .at()
const bool has_sub = pattern_map.count(sub_m) > 0;          // optional node -> .count()
```

Use `.at()` for mandatory entries. `operator[]` and `count()`-guarded lookups on mandatory labels hide pattern/callback mismatches.

### Trust the matcher

Do not re-verify what the pattern already guarantees:

- no `as_type_ptr` + null check on a node matched by `wrap_type<T>` (unless `T`-specific API is actually used);
- no input-count checks when `validate_inputs_count` or the op constructor already enforced them;
- no shape/type re-checks that duplicate a predicate.

### Fail fast on violated invariants

If a condition can only be false because the matcher or the pipeline is broken, assert — do not `return false`. A silent `return false` produces a transformation that "did nothing", which is the hardest class of transformation defect to diagnose. Assertions also keep static analyzers (Coverity) quiet about the unchecked cast.

```cpp
// Avoid
auto swish = ov::as_type_ptr<v4::Swish>(pattern_map[swish_m].get_node_shared_ptr());
if (!swish)
    return false;

// Prefer
const auto swish = ov::as_type_ptr<v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
OPENVINO_ASSERT(swish, "MyFusion: matched node is expected to be v4::Swish");
```

`return false` is reserved for "the pattern matched, but this instance is legitimately not transformable" — and even then, prefer moving the condition into the pattern (see [Pattern matching](#pattern-matching)).

### Compute constants with OV ops, not raw buffers

To compare or derive constant values (FakeQuantize ranges, scales, zero points), build a small OV subgraph and fold it instead of iterating raw data. The opset implementation handles broadcasting, mixed precisions and corner cases for free, and the cost is paid once at compile time.

```cpp
auto reshape = std::make_shared<v1::Reshape>(bias, new_shape, false);
auto folded  = ov::util::get_constant_from_source(reshape);
auto new_bias = folded ? folded->output(0) : reshape->output(0);
```

This also simplifies the matcher: build the node unconditionally, fold it, and use the folded constant if folding succeeded — instead of separating constant and non-constant cases in the pattern.

## Modifying the graph

### Use the canonical replacement helpers

| Task | Helper |
|------|--------|
| Replace one output by another, preserving tensor names | `ov::replace_output_update_name(old_out, new_out)` |
| Replace a node by a new one | `ov::replace_node(old_node, new_node)` |
| Rebuild a node with different inputs | `node->clone_with_new_inputs({...})` |
| Rewire a single input edge | `node->input(i).replace_source_output(new_out)` |

Prefer `replace_node` with a cloned node over a sequence of `replace_source_output` calls: the latter is easy to get partially wrong and loses the original node's identity.

### Preserve runtime info and friendly names

```cpp
ov::copy_runtime_info({old_node_1, old_node_2}, {new_node_1, new_node_2});
new_node->set_friendly_name(old_node->get_friendly_name());
```

`copy_runtime_info` must be called for every created node. Runtime info carries precision markup, fusing hints and provenance that downstream passes and plugins rely on; losing it changes behavior silently and is caught by `TransformationTestsF`'s `RUNTIME_KEYS` check.

Attributes already applied by dedicated markup passes (`keep_const_precision`, `DisableFP16Compression`, precision attributes) must not be re-applied ad hoc inside an unrelated transformation — fix the markup pass instead.

### Recreate `TypeRelaxed` nodes by cloning

A `TypeRelaxed<T>` node carries `_input_data_types` / `_output_data_types`. Constructing a fresh `TypeRelaxed<T>` and patching only the output precision loses the input configuration and forces spurious `Convert` insertion:

```cpp
// Avoid
auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(parent, scales);
NetworkHelper::setOutDataPrecisionForTypeRelaxed(multiply, dq.multiply->get_output_element_type(0));

// Prefer — clones both input and output type configuration
auto multiply = dq.multiply->clone_with_new_inputs({parent, scales});
```

### Let the framework clean up

Dead consumers left behind by a rewrite are removed by the next `Validate` run. Do not add manual clean-up code.

### Respect the operation specification

A rewrite must match the documented semantics of the operations involved. For example, `FakeQuantize` output shape always equals its data-input shape, so an eltwise operation that relies on broadcasting to a constant's shape cannot be fused into it — no matter what the eltwise type is. Gate the transformation on the *semantic* property, never on a proxy such as an operation type name or the presence of a neighboring node.

## Low precision specifics

When rebuilding dequantization arithmetic:

- compute in the precision that cannot overflow — typically the multiply-constant precision, which is always floating point — not in the subtract-constant precision, which may be low precision;
- align precisions with `foldConvert` / `ov::fundamental_type_for` instead of hardcoding them;
- express limits with `std::numeric_limits<T>` (e.g. `std::numeric_limits<ov::float16>::max()`), never as literals;
- validate that constant values are representable in the target data type before transforming (e.g. a `Pad` value outside the `u8` range must block the transformation);
- consider inserting `Round` before `Convert` when converting float values to an integer bias.

Additional LPT-specific guidance: [low_precision_transformations tests README](../../low_precision_transformations/tests/README.md).

## Internal operations (`ov_ops`)

- Do not expose a constructor that leaves the op unconfigured, nor setters that mutate its configuration after construction. Configuration is passed once, at construction.
- Override `visit_attributes` for any op with custom attributes. It is required for `ov::Model` serialization/deserialization (model caching) and enables generic passes such as `ov::pass::SharedOpOptimization`.
- `validate_and_infer_types` must reject configurations that the specification or the implementations do not support, with an actionable message. Accepting an invalid configuration and silently ignoring part of it is a defect.
- Absent optional inputs follow the shared convention: `element::dynamic` element type with an empty shape.
- Use semantic types: `bool` for flags, `std::optional<T>` instead of sentinel values, no default argument values that are used at a single call site.
- Type-propagation tests belong in [src/core/tests/type_prop/](../../../core/tests/type_prop/).

## Registering the pass in a pipeline

- `InitNodeInfo` must remain the first pass of a pipeline; `Validate` should remain the last one — its cost is low and it catches precision/shape-inference breakage introduced by the pipeline.
- A pass that is a prerequisite of another pass must be registered in **every** pipeline where that pass runs: `CommonOptimizations`, `MOCTransformations`, the CPU/GPU/NPU pipelines, and the plugin FQ-stripping pipelines.
- Before adding a registration, check whether the pass already runs earlier in the same pipeline.
- Changing the position of a pass must be assessed against existing consumers. Moving SDPA decomposition earlier, for example, breaks `SDPAWithKVCache` fusion in the CPU plugin and therefore all LLM scenarios.
- Prefer reordering the pipeline over adding a pass that compensates for the current order (or over relaxing a pattern to tolerate a node that an earlier pass would have removed).
- Keep plugin pipeline files from growing without bound: extract sets of related callbacks into their own translation unit.

## Performance considerations

Transformations run at compile time, but they determine the executed graph:

- **Quantized paths.** Replacing a fused quantized primitive with a decomposed sequence can be a regression even when it helps the compressed-weights case. Gate the transformation on the model not being quantized when that applies.
- **Plugin post-op fusion.** Inserting a `Transpose`/`Reshape` between a layer and its bias or activation breaks plugin fusing. Place the inserted node so the fusable chain stays intact (e.g. after the bias, not between bias and MatMul).
- **No-op rewrites.** A transformation should not fire when its output is equivalent to its input — for example, inserting a transpose when `H * W == 1`.
- **Over-broad markup.** A generic "disable precision conversion" pass may cover far more subgraphs than required. Prefer targeted per-pattern passes.
- **Memory.** Keep the weight-decompression subgraph unfolded where that is the memory-efficient form, and fold constants deliberately rather than accidentally materializing large tensors.

## Documenting the pass

Each pass header states: what the pass does, why it is needed, and a scheme of the matched and produced subgraph with optional nodes marked.

```cpp
/**
 * @ingroup ov_transformation_common_api
 * @brief Moves the dequantization Multiply from the Convolution output to its weights.
 *
 * Needed to avoid f16 overflow in the Convolution accumulator on ARM/ACL.
 *
 * Before:                          After:
 *
 *   Conv(u8 act, i8 w)               Conv(u8 act, i8 w * dq_scale)
 *          │
 *   Multiply(dq_scale)
 *
 * The transformation is skipped when the scale is per-channel on a non-output axis,
 * since folding it into the weights would change the result.
 */
```

Rules for comments:

- Verify every comment against the code. A comment that contradicts the expression it documents, or carries a factually wrong justification, is a defect.
- Delete comments that merely restate a symbolic predicate — once `shape_matches("?, ?, 1, 1")` is in the pattern, `// 1x1 spatial dims` adds nothing.
- Document non-obvious numeric criteria (tolerance formulas, relative vs absolute comparisons) with at least a pseudo-formula.
- Avoid enumerating specific operation types in a description when the list may grow ("value-preserving ops" instead of "Reshape, Squeeze, Unsqueeze").
- Avoid mentioning exact precisions when the pass is parameterized by a precision list.

## Naming

- Pass and variable names describe *what* is done and *when*, not the model that motivated the change. `FallbackUnsupportedLPConvToFP16` beats `ConvertConvDQScales`; `DisableBF16CompForLtxVideoRopePattern` beats `MarkSinCosInputsPrecision`.
- Rename when behavior broadens: a `gptoss_gemma3_mask` variable that now also covers gemma4 becomes `mask`.
- Align the suffix with existing passes: `*Fusion`, `*Decomposition`, `*Elimination`, `Convert*To*`.
- Pattern node labels end with `_m` (`conv_m`, `weights_m`) — the convention that distinguishes pattern nodes from matched nodes in the callback.
- Name pattern dimensions after their meaning (`hidden_in`, `seq_len`), not `?`.

## Testing

Every behavioral change ships with tests. The rules are in [Writing transformation tests](./writing_tests.md); the essentials:

- use `TransformationTestsF` with `model` / `manager` / `model_ref` and let `FunctionsComparator` verify the result — never manual node counting;
- leave `model_ref` unset for negative cases;
- add a regression test reproducing the reported scenario for every bug fix, ideally built from the failing model's subgraph;
- parametrize instead of copy-pasting instances, and reuse shared model builders;
- cover corner cases explicitly: dynamic shapes, per-channel vs per-tensor, missing optional inputs, unsupported precisions;
- never weaken a test to make it pass.

Contributors using an AI agent can apply the `ov-transformation-tests` skill to align tests with these rules.

## Reusable utilities

Search for an existing utility before adding a helper or a pass. Frequently missed candidates:

| Need | Existing utility |
|------|------------------|
| Fuse identical sibling operations (horizontal fusion) | `ov::pass::SharedOpOptimization` |
| Walk a producer / consumer chain | `ov::op::util::visit_path`, `ov::op::util::visit_path_forward` |
| Replace one output by another, keeping names | `ov::replace_output_update_name` |
| Fold a subgraph to a constant if possible | `ov::util::get_constant_from_source`, `ov::op::util::clone_try_fold` |
| Compare two constants | `ov::compare_constants` |
| Compare FakeQuantize parameters | `ov::op::util::have_same_fake_quantize_params` |
| Multi-type check | `ov::is_type_any_of<T1, T2>(node)` |
| Compare possibly-dynamic dimensions | `ov::symbol::util::dims_are_equal` |
| Extract a subgraph into a standalone model (debug builds) | `ov::util::extract_subgraph` |

If an existing utility is insufficient, extend it — do not fork it. Duplicated non-trivial logic diverges, and fixes then land in only one copy.

## What NOT to do

- Do **not** put matching conditions in the callback when a predicate can express them
- Do **not** write a custom predicate lambda before checking the [predicate reference](#predicate-reference)
- Do **not** hand-roll `Or` chains where `optional<>` or `operator|` applies
- Do **not** re-check what the matcher guarantees (`as_type_ptr` + null check, input-count checks, duplicated shape checks)
- Do **not** use `operator[]` or `count()`-guarded access for mandatory pattern-map entries
- Do **not** `return false` on a broken invariant — assert
- Do **not** traverse the graph manually in a callback or a plugin pipeline when a pattern or `visit_path` fits
- Do **not** create a node without `copy_runtime_info`, or drop friendly names
- Do **not** construct a fresh `TypeRelaxed<T>` where `clone_with_new_inputs` is meant
- Do **not** clean up dead nodes manually — `Validate` does it
- Do **not** gate a transformation on a proxy signal (op type name, neighboring node, model-specific shape)
- Do **not** hardcode precisions, thresholds or model-specific values — parameterize
- Do **not** add a restriction (consumer count, per-tensor, const-only, static-shape-only) without a documented reason
- Do **not** duplicate a device-agnostic pass per plugin — parameterize one common pass
- Do **not** grow a pass with variant branches — split into matchers under a `GraphRewrite`
- Do **not** mix unrelated changes into the PR; split refactors into follow-ups

## See also

- [Writing transformation tests](./writing_tests.md)
- [Matcher logging](./debug_capabilities/matcher_logging.md)
- [Transformation profiling](./debug_capabilities/transformation_profiling.md)
- [Transformations documentation index](./README.md)
- [OpenVINO contribution guide](/CONTRIBUTING.md)

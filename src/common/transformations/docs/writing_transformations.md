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
- [Documenting the pass](#documenting-the-pass)
- [Testing](#testing)
- [Reusable utilities](#reusable-utilities)
- [What NOT to do](#what-not-to-do)

## Where transformations live

| Kind | Location |
|------|----------|
| Device-agnostic optimizations | [src/common/transformations/src/transformations/common_optimizations/](../src/transformations/common_optimizations/) |
| Opset conversions / decompositions | [src/common/transformations/src/transformations/op_conversions/](../src/transformations/op_conversions/) |
| FP16/BF16 compression markup | [src/common/transformations/src/transformations/fp16_compression/](../src/transformations/fp16_compression/) |
| Reusable pattern blocks | [src/common/transformations/include/transformations/pattern_blocks/](../include/transformations/pattern_blocks/) |
| Plugin-specific passes | `src/plugins/<plugin>/src/transformations/` |

A pass that is beneficial regardless of the backend belongs in common transformations, **not** duplicated per plugin. Express device-specific behavior as a pass parameter (e.g. a list of supported precisions) instead of forking the pass.

New files are not picked up automatically: list the `.cpp` in [src/sources.cmake](../src/sources.cmake) and the `.hpp` in [include/sources.cmake](../include/sources.cmake).

## Quick checklist

1. Pick the right base class: `MatcherPass` for local rewrites, `ModelPass` when nodes outside the matched pattern are modified
2. Put **every** applicability condition into the pattern, not into the callback
3. Prefer built-in predicates (`shape_matches`, `type_matches_any`, `attrs_match`, …) over custom lambdas
4. Access mandatory pattern nodes with `pattern_map.at(label)`; assert on invariants instead of `return false`
5. Call `copy_runtime_info` for every created node and preserve friendly names
6. Try to solve the general problem —  topology-specific proxies and hardcoded model-specific values are permissible with strong argumentation only
7. Justify every restriction (consumer count, per-tensor only, const-only, static-shape-only)
8. Describe what the pass does and why in the header, with a subgraph scheme
9. Add tests per [Writing transformation tests](./writing_tests.md)

## Choosing the pass type

| Pass type | Use when |
|-----------|----------|
| `ov::pass::MatcherPass` | The rewrite touches only nodes contained in the matched pattern. This is the default choice. |
| `ov::pass::ModelPass` | The rewrite modifies nodes outside the pattern (consumers of the match root, distant producers, model inputs/outputs/sinks), or needs global model state. |
| `ov::pass::GraphRewrite` | Several independent matchers should run in one graph traversal. Prefer this over branching on variants inside a single callback. |
| `ov::pass::BackwardGraphRewrite` | The matchers require bottom-up traversal, most commonly because the pattern root is optional. |

Several rules follow from this table:

- **A pass has a single responsibility.** Long `if (case_1) … else if (case_2) …` chains inside one callback signal that the pass is doing two jobs — split them into separate matchers registered in one `GraphRewrite`, sharing a common helper.
- **Register the matcher in a `BackwardGraphRewrite` when the pattern root is optional.** `GraphRewrite` visits nodes in topological order, so a matcher whose root is an `optional<...>` node fires on the shorter variant as soon as the optional node's producer is reached, and the full pattern is never tried. Backward traversal reaches the optional tail first and matches the complete pattern. Encapsulate this in the pass itself — expose the matcher as a standalone `MatcherPass` and register it inside a `BackwardGraphRewrite` container.
- Prefer composing existing generic passes over re-implementing their effect inline.

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
#include "openvino/op/convolution.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::MyFusion::MyFusion() {
    MATCHER_SCOPE(MyFusion);   // defines `matcher_name`, enables conditional compilation

    using namespace ov::pass::pattern;

    auto weights_m = any_input(type_matches(element::i8));
    auto conv_m    = wrap_type<ov::op::v1::Convolution>({any_input(), weights_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto conv = pattern_map.at(conv_m).get_node_shared_ptr();
        if (transformation_callback(conv)) {
            return false;
        }
        // ... build the replacement
        return true;
    };

    auto m = std::make_shared<Matcher>(conv_m, matcher_name);
    register_matcher(m, callback);
}
```

`MATCHER_SCOPE` / `RUN_ON_MODEL_SCOPE` are mandatory: they provide the ITT domain name used by transformation profiling and allow the pass to be excluded from conditionally-compiled builds.

Use `OV_CAPTURE_CPY_AND_THIS` in the callback lambda capture list instead of `[=]` or `[&]`.

A few mechanics are worth remembering:

- **One matcher per `MatcherPass`.** `register_matcher` is called exactly once. Several matchers belong in a `GraphRewrite`.
- **The callback return value is meaningful.** Return `true` when the matched graph was rewritten in any way — no other matcher will be tried on that root. Return `false` when the graph was left untouched.
- **Nodes created by the callback are not re-matched by default.** If they should be picked up by the other matchers of the same `GraphRewrite`, report them with `register_new_node` (in topological order).
- **A pass must be idempotent.** The same matcher can be applied again — to nodes the callback created, on a repeated run of the pipeline, or from another container. Running the pass twice must leave the graph unchanged the second time.

## Pattern matching

**Every condition that decides whether the pass applies should be expressed in the pattern whenever possible** — operation type, rank, shape, element type, attribute values, whether an input has to be a `Constant`, whether a shape has to be static. Sometimes a condition cannot be expressed in the pattern; only in those cases callback-based checks are applicable. The callback is otherwise responsible only for *building the replacement*.

Conditions in the pattern are self-documenting, composable, enforced by the matcher (so the callback needs no defensive code), and — crucially — visible in [matcher logs](./debug_capabilities/matcher_logging.md), which turns "why didn't my pass fire?" into a log-reading exercise instead of a debugging session.

Avoid:

```cpp
auto add_m = wrap_type<v1::Add>();

ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
    auto add = ov::as_type_ptr<v1::Add>(m.get_match_root());
    if (!add)
        return false;
    const auto& data_shape = add->get_input_partial_shape(0);
    if (data_shape.is_dynamic() || data_shape.rank().get_length() != 2)
        return false;
    const auto data_type = add->get_input_element_type(0);
    if (data_type != element::u8 && data_type != element::i8)
        return false;
    // ... actual rewrite
};
```

Prefer:

```cpp
auto data_m = any_input(has_static_shape() && rank_equals(2) && type_matches_any({element::u8, element::i8}));
auto add_m  = wrap_type<v1::Add>({data_m, any_input()});

ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
    const auto& pattern_map = m.get_pattern_value_map();
    const auto data = pattern_map.at(data_m);
    // ... actual rewrite only
};
```

The same rule applies to plugin pipelines: manual model traversal used to decide whether to register a pass should instead become a pass parameter plus a pattern predicate.

A pattern is a single-rooted graph: the node passed to `Matcher` is the root, and pattern nodes that neither feed the root nor are the root itself do not participate in matching at all. A predicate attached to such a dangling node is silently ignored.

### Predicate reference

Here are some commonly used predicates:

| Need | Predicate |
|------|-----------|
| Rank only | `rank_equals(3)`, `rank_more_than(1)` |
| Element type | `type_matches(element::i8)`, `type_matches_any({element::u8, element::i8})` |
| Node attributes | `attrs_match({{"mode", "numpy"}})`, or the `attrs` argument of `wrap_type` |
| Static shape / rank | `has_static_shape()`, `has_static_rank()`, `has_static_dim(pos)`, `has_static_dims({...})` |
| Consumers | `consumers_count(1)`, `consumers_more_than(1)` |

Predicates can be combined with `&&`, `||` and `!`.

For the complete list, see [pattern.hpp](../../../core/include/openvino/pass/pattern/op/pattern.hpp).
If the existing predicates, including their combinations, cannot express the condition, use a custom `pattern::Predicate` lambda.

Node builders:

| Need | Builder |
|------|---------|
| Node of one of several types | `wrap_type<v0::Sin, v0::Cos>({input})` |
| Any node with a predicate | `any_input(pred)` |
| Any `Constant` | `wrap_const()` |
| Optional node in the chain | `optional<v1::Subtract>({input, any_input()})` |
| Alternatives | `a \| b` |
| Multi-output node with explicitly indexed outputs | `wrap_type_strict_index<v1::Split>({input, axis})->output(i)` |

**Note:** If `wrap_type_strict_index` is used, access a matched multi-output pattern node through `Matcher::get_pattern_map()`, not `Matcher::get_pattern_value_map()`, which stores a matched output rather than the node.

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

**Note:** Use only the 0th input as a 'data' input for optional. So `optional<T>({a, b})` can only bypass to `a`.

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

**Important note:** symbolic predicates usage require symbols to have been propagated through the model first. Ensure `SymbolicOptimizations(false)` runs before the pass that uses symbols:

```cpp
ov::pass::Manager manager;
manager.register_pass<ov::pass::SymbolicOptimizations>(false);
manager.register_pass<MyPass>();
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

### Fail fast on violated invariants

Use assertions for conditions that can only fail if the matcher or pipeline is broken. For example, a type mismatch for a node matched by a single-type `wrap_type<T>` violates an invariant; asserting instead of returning `false` avoids silently skipping the transformation and keeps static analyzers (Coverity) quiet about unchecked casts.

```cpp
// Matcher
auto swish_m = wrap_type<v4::Swish>();

// Avoid
auto swish = ov::as_type_ptr<v4::Swish>(pattern_map[swish_m].get_node_shared_ptr());
if (!swish)
    return false;

// Prefer
const auto swish = ov::as_type_ptr<v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
OPENVINO_ASSERT(swish, "MyFusion: matched node is expected to be v4::Swish");
```

Use `return false` when the matched instance is legitimately not transformable:

```cpp
auto new_weights = fuse_const_to_weights(matmul, weights, mul_const);
// Fusion was not successfull
if (!new_weights)
    return false;
```

### Trust the matcher

Do not re-verify what the pattern already guarantees:

- do not return false if `as_type_ptr` returns nullptr for a node matched by `wrap_type<T>`, use assert instead;
- no input-count checks when `validate_inputs_count` or the op constructor already enforced them;
- no shape/type re-checks that duplicate a predicate.

### Compute constants with OV ops, not raw buffers

To compare or derive constant values (FakeQuantize ranges, scales, zero points), build a small OV subgraph and fold it instead of iterating raw data. The opset implementation handles nuances such as broadcasting authomatically, and the cost is paid once at model compile time.

```cpp
auto diff = std::make_shared<ov::op::v1::Subtract>(val1, val2);
auto abs_diff = std::make_shared<ov::op::v0::Abs>(diff);
auto eps = ov::op::v0::Constant::create(val1.get_element_type(), {}, {1e-6f});
auto is_less = ov::util::get_constant_from_source(std::make_shared<ov::op::v1::Less>(abs_diff, eps));
```

This also simplifies the matcher: build the node unconditionally, fold it, and use the folded constant if folding succeeded — instead of separating constant and non-constant cases in the pattern.

## Modifying the graph

### Use the canonical replacement helpers

| Task | Helper |
|------|--------|
| Eliminate a node, reconnecting consumers to its input | `ov::replace_output_update_name(old_out, new_out)` |
| Replace a node by a new one | `ov::replace_node(old_node, new_node)` |
| Rebuild a node with different inputs | `node->clone_with_new_inputs({...})` |
| Rewire a single input edge | `node->input(i).replace_source_output(new_out)` |

Prefer `replace_node` with a cloned node over a sequence of `replace_source_output` calls: the latter is easy to get partially wrong and loses the original node's identity. Note that `replace_node` requires both nodes to have the same number of output ports and throws otherwise.

`replace_output_update_name` copies the runtime info of the eliminated node onto the replacement and keeps its friendly name when the output feeds a `Result`. It returns `false` when that Result-facing name cannot be preserved safely: if either node has multiple outputs, the replacement is a `Parameter`, or the replacement already has a `Result` consumer. Check the return value instead of assuming the node is gone.

**Note:** Replacing an output redirects all its consumers, including those not represented in the matcher. If the rewrite is not valid for such additional consumers, use `consumers_count(...)` in the pattern to require that every consumer of the original output is represented in the matcher. Rewiring one input edge changes no other consumers.

### Preserve runtime info and friendly names

Runtime info carries precision markup, fusing hints and provenance that downstream passes and plugins rely on. It is **not** propagated automatically, and losing it changes behavior silently — `TransformationTestsF`'s `RUNTIME_KEYS` check exists to catch exactly that.

```cpp
ov::copy_runtime_info(transpose, reshape);          // 1:1  — node replaced by a node
ov::copy_runtime_info(div, {pow, mul});             // 1:N  — node replaced by a subgraph
ov::copy_runtime_info({conv, bias}, {conv_fused});  // N:1  — subgraph fused into a node
ov::copy_runtime_info({a, b, c}, {e, f});           // N:M  — anything else

new_node->set_friendly_name(old_node->get_friendly_name());
```

`copy_runtime_info` overwrites destination attributes whose keys are also present in the sources. To let the destination's own attributes participate in the merge instead of being overwritten, list the destination among the sources: `copy_runtime_info({a, b, c}, {a, b})`.
This merge applies only to `ov::RuntimeAttribute` values: keys with multiple non-attribute values or an empty `merge()` result are dropped, and `assign_runtime_info` preserves only the destination's `opset` key.

When a subgraph is replaced by another subgraph, the original friendly name goes to the **last** node of the replacement.

Attributes already applied by dedicated markup passes (`keep_const_precision`, `DisableFP16Compression`, precision attributes) must not be re-applied ad hoc inside an unrelated transformation — fix the markup pass instead.

### Fold the constant subgraphs you create if possible

When a transformation creates a foldable subgraph, fold it immediately rather than relying on a later `ov::pass::ConstantFolding` pass: use `ov::op::util::make_try_fold<T>(...)` to construct an op and keep its folded result when possible (otherwise it returns the op), or `ov::util::get_constant_from_source(output)` when a Constant is needed (it returns nullptr if folding is not possible). This avoids carrying foldable nodes through the rest of the pipeline and is more profitable from model compilation time perspective.

## Documenting the pass

Each pass header states: what the pass does, why it is needed, a before/after scheme of the matched and produced subgraph with optional nodes marked, and the conditions under which the rewrite is applied. An excerpt from [broadcast_matmul_fusion.hpp](../include/transformations/common_optimizations/broadcast_matmul_fusion.hpp):

```cpp
/**
 * @ingroup ov_transformation_common_api
 * @brief Removes a redundant Broadcast that expands one MatMul input's batch dimensions.
 *
 * Matches the Data -> Broadcast -> MatMul pattern, with the Broadcast on either MatMul
 * input and Data being an arbitrary input (not necessarily a Constant). MatMul broadcasts
 * the batch (leading) dimensions of its operands implicitly, so an explicit Broadcast that
 * only expands those dimensions is redundant.
 *
 * Before:                          After:
 *
 *     Data          Other              Data          Other
 *       │             │                  │             │
 *   ┌───┴─────┐       │                  │             │
 *   │Broadcast│       │                  │             │
 *   └───┬─────┘       │                  │             │
 *       │             │                  │             │
 *       └──────┬──────┘                  └──────┬──────┘
 *           ┌──┴───┐                         ┌──┴───┐
 *           │MatMul│                         │MatMul│
 *           └──────┘                         └──────┘
 *
 * The Broadcast is removed only when it does not change the MatMul result:
 *  - the matrix (last two) dimensions are left intact by the Broadcast;
 *  - for every expanded batch dimension, the other MatMul operand carries the same
 *    dimension, proven equal by static value or by shape symbol; an unlabeled dynamic
 *    dimension is never assumed compatible, since that could hide a runtime batch mismatch
 *    the Broadcast would have rejected.
 */
```

Rules for comments:

- Verify every comment against the code. A comment that contradicts the expression it documents, or carries a factually wrong justification, is a defect.
- Delete comments that merely restate a symbolic predicate — once `shape_matches("?, ?, 1, 1")` is in the pattern, `// 1x1 spatial dims` adds nothing.
- Document non-obvious numeric criteria (tolerance formulas, relative vs absolute comparisons) with at least a pseudo-formula.
- For non-trivial transformation logic, add shape annotations when they clarify how shapes change through the transformation.
- Avoid enumerating specific operation types in a description when the list may grow ("value-preserving ops" instead of "Reshape, Squeeze, Unsqueeze").
- Avoid mentioning exact precisions when the pass is parameterized by a precision list.

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

Search for an existing utility before adding a helper or a pass. Start with these headers:

- [transformations/utils/utils.hpp](../include/transformations/utils/utils.hpp) — the most frequently needed `ov::op::util` helpers: node creation and folding, graph traversal (`visit_path`, `visit_path_forward`), constant and FakeQuantize comparison, shape checks.
- [openvino/core/graph_util.hpp](../../../core/include/openvino/core/graph_util.hpp) — graph modification and inspection: `replace_node`, `replace_output_update_name`, `compare_constants`, `topological_sort`.
- [openvino/core/rt_info.hpp](../../../core/include/openvino/core/rt_info.hpp) — `copy_runtime_info` overloads.
- [openvino/core/validation_util.hpp](../../../core/dev_api/openvino/core/validation_util.hpp) — `get_constant_from_source`, axis and shape normalization.
- [openvino/core/type.hpp](../../../core/include/openvino/core/type.hpp) — `is_type`, `is_type_any_of`, `as_type_ptr`.
- [transformations/symbolic_transformations/utils.hpp](../include/transformations/symbolic_transformations/utils.hpp) — symbol-aware dimension and shape comparison for dynamic shapes.

If an existing utility is insufficient, extend it — do not fork it. Duplicated non-trivial logic diverges, and fixes then land in only one copy.

## What NOT to do

- Do **not** put matching conditions in the callback when a predicate can express them
- Do **not** write a custom predicate lambda before checking the [predicate reference](#predicate-reference)
- Do **not** hand-roll `Or` chains where `optional<>` or `operator|` applies
- Do **not** re-check what the matcher guarantees (mandatory nodes existence, input-count checks, duplicated shape checks)
- Do **not** `return false` on a broken invariant — assert
- Do **not** traverse the graph manually in a callback or a plugin pipeline when a pattern or `visit_path` fits
- Do **not** modify nodes that come after the match root in topological order from a `MatcherPass` callback — use a `ModelPass`
- Do **not** pass a `shared_ptr<Node>` as an input when the producer type is unknown or has several outputs — pass the explicit output port
- Do **not** target an older opset unless the pass is a downgrade transformation
- Do **not** create a node without `copy_runtime_info`, or drop friendly names
- Do **not** clean up dead nodes manually without a strong justification — `Validate` does it
- Do **not** add a restriction (consumer count, per-tensor, const-only, static-shape-only) without reason
- Do **not** duplicate a device-agnostic pass per plugin — parameterize one common pass
- Do **not** grow a pass with variant branches — split into matchers under a `GraphRewrite`

## See also

- [Writing transformation tests](./writing_tests.md)
- [Matcher logging](./debug_capabilities/matcher_logging.md)
- [Transformation profiling](./debug_capabilities/transformation_profiling.md)
- [Transformations documentation index](./README.md)
- [OpenVINO contribution guide](/CONTRIBUTING.md)

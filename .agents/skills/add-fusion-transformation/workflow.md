# Skill: Add OpenVINO Fusion Transformation (End-to-End)

> **Upstream skill spec:** `skills/add-fusion-transformation/SKILL.md`
> This file is the Copilot-ready, agent-executable version of that specification.

## Purpose

End-to-end workflow for implementing a new **pattern-based graph fusion transformation**
in OpenVINO. Covers: problem framing → design → implementation → registration →
unit tests → patch generation.

Based on PR #40 ("Skill Workflow for OpenVINO Fusion transformation") and the
OpenVINO transformation API patterns in `src/common/transformations/`.

---

## Scope

This workflow applies to OpenVINO transformation development in `openvino/` for:

- Pattern-based graph fusion
- Subgraph replacement with:
  - a single fused op, or
  - several ops preserving semantics with lower runtime overhead

The workflow covers:

1. Pattern design and safety guards
2. Pass implementation and registration
3. Runtime correctness constraints (shape/type/friendly name/rt_info)
4. Unit and functional regression tests

---

## Architecture

```
Model Graph → Pattern Matcher Pass → Subgraph Replacement → Validation/Inference Tests
```

---

## Repository Layout

```
openvino/
├─ src/common/transformations/
│  ├─ include/transformations/
│  │  └─ <domain>/<fusion_name>.hpp
│  ├─ src/transformations/
│  │  └─ <domain>/<fusion_name>.cpp
│  ├─ src/transformations/common_optimizations/
│  │  └─ common_optimizations.cpp   (or relevant pipeline file)
│  └─ tests/
│     ├─ common_optimizations/
│     │  └─ <fusion_name>.cpp
│     └─ utils/
└─ tests/
   └─ layer_tests/ or functional/    (if user-visible behavior/perf contract is affected)
```

---

## When to invoke

- Transformation Agent, whenever a new fusion pass needs to be created from scratch
- `patch_type=transformation` shortcut path (incoming patch already exists — apply,
  validate, then follow Steps 4–6 for testing and registration)

---

## Steps

### Step 1: Confirm the fusion does not already exist

- Search for equivalent matcher patterns and passes (`grep -r "MatcherPass\|register_pass" src/common/transformations/`).
- If a similar fusion exists, extend it instead of creating a duplicate pass.
- Reuse existing helper utilities and pattern predicates where possible.

---

### Step 2: Define fusion contract and understand the problem

Before writing any code:

1. **Read the op spec** (from Core OpSpec Agent or issue comment) to confirm:
   - What does the new fused op do?
   - What sub-graph does it replace?
   - Are there accuracy or performance requirements?

2. **Confirm the fusion makes sense at graph level:**
   - The pattern must be composable (inputs/outputs align exactly)
   - No side-effecting ops in the middle of the pattern
   - The replacement preserves dtype and broadcast semantics

3. **Specify semantic invariants** documented before coding:
   - Output values, element types, ranks/dynamic-shape compatibility
   - Runtime info / friendly names preserved
   - Explicit no-fuse conditions (negative guards)

4. **Choose the pass type** using `openvino_transformation_analysis` skill.

### Step 3: Design the Fusion Pattern

Produce a before/after diagram:

```
BEFORE:
  x ──► MatMul(transB=true) ──► Add ──► gate1
           ▲                       ▲
  W ───────┘          bias ────────┘

AFTER:
  x ──► FusedLinear(W, bias) ──► gate1
```

Write down:
- Matching conditions (e.g., `transB=true` required, `bias` must be 1-D Constant)
- Output consumer constraints (e.g., `Add` must have exactly one consumer)
- Edge case exclusions (e.g., no fusion when weights are FP16 quantised)

### Step 4: Implement Header and Source

Follow **`openvino_transformation_implementation`** skill exactly.

Checklist after writing:
- [ ] `OPENVINO_RTTI("ClassName", "0")` present
- [ ] `MATCHER_SCOPE` macro used (sets `matcher_name`)
- [ ] `ov::copy_runtime_info` called before `ov::replace_node`
- [ ] `friendly_name` copied from the root node of the replaced sub-graph
- [ ] No raw `delete` or manual output re-wiring

### Step 5: Register the Pass

Determine registration location from the analysis output.

For **common_optimizations**:
```cpp
// Find the section that groups related fusion passes (e.g., linear layer fusions)
// Insert FuseMyOp alphabetically/logically within that group
ADD_MATCHER(manager, FuseMyOp)
```

Verify the registration order does not break existing passes:
- Constant folding must run BEFORE pattern-matching passes that require constant inputs
- Layout-changing passes must run AFTER all arithmetic fusions

### Step 6: Write Unit Tests

Location: `src/common/transformations/tests/<domain>/`

Required test file: `test_<pass_name>.cpp`

```cpp
#include <gtest/gtest.h>
#include "transformations/<domain>/<pass_name>.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

using namespace ov;

// --- Test 1: Positive — pattern fires correctly ---
TEST_F(TransformationTestsF, FuseMyOp_Basic) {
    {  // "model" block — the input graph
        auto input   = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 768});
        auto weights = op::v0::Constant::create(element::f32, Shape{768, 3072}, {0.1f});
        auto bias    = op::v0::Constant::create(element::f32, Shape{3072}, {0.0f});
        auto mm      = std::make_shared<op::v0::MatMul>(input, weights, false, true);
        auto add     = std::make_shared<op::v1::Add>(mm, bias);
        auto result  = std::make_shared<op::v0::Result>(add);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

        manager.register_pass<pass::FuseMyOp>();
    }
    {  // "model_ref" block — the expected output graph
        auto input_r   = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 768});
        auto weights_r = op::v0::Constant::create(element::f32, Shape{768, 3072}, {0.1f});
        auto bias_r    = op::v0::Constant::create(element::f32, Shape{3072}, {0.0f});
        auto fused_r   = std::make_shared<op::internal::MyFusedOp>(input_r, weights_r, bias_r);
        auto result_r  = std::make_shared<op::v0::Result>(fused_r);
        model_ref = std::make_shared<Model>(ResultVector{result_r}, ParameterVector{input_r});
    }
}

// --- Test 2: Negative — non-constant weights, pass must NOT fire ---
TEST_F(TransformationTestsF, FuseMyOp_Negative_DynamicWeights) {
    {
        auto input   = std::make_shared<op::v0::Parameter>(element::f32, Shape{2, 768});
        auto weights = std::make_shared<op::v0::Parameter>(element::f32, Shape{768, 3072});
        auto bias    = op::v0::Constant::create(element::f32, Shape{3072}, {0.0f});
        auto mm      = std::make_shared<op::v0::MatMul>(input, weights, false, true);
        auto add     = std::make_shared<op::v1::Add>(mm, bias);
        auto result  = std::make_shared<op::v0::Result>(add);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input, weights});

        manager.register_pass<pass::FuseMyOp>();
    }
    // model_ref not set → TransformationTestsF checks model is UNCHANGED
}

// --- Test 3: Output type preserved ---
TEST_F(TransformationTestsF, FuseMyOp_OutputTypePreserved) {
    {
        auto input   = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 768});
        auto weights = op::v0::Constant::create(element::f16, Shape{768, 3072}, {0.1f});
        auto bias    = op::v0::Constant::create(element::f16, Shape{3072}, {0.0f});
        auto mm      = std::make_shared<op::v0::MatMul>(input, weights, false, true);
        auto add     = std::make_shared<op::v1::Add>(mm, bias);
        auto result  = std::make_shared<op::v0::Result>(add);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{input});

        manager.register_pass<pass::FuseMyOp>();
    }
    {
        auto input_r   = std::make_shared<op::v0::Parameter>(element::f16, Shape{2, 768});
        auto weights_r = op::v0::Constant::create(element::f16, Shape{768, 3072}, {0.1f});
        auto bias_r    = op::v0::Constant::create(element::f16, Shape{3072}, {0.0f});
        auto fused_r   = std::make_shared<op::internal::MyFusedOp>(input_r, weights_r, bias_r);
        model_ref = std::make_shared<Model>(ResultVector{std::make_shared<op::v0::Result>(fused_r)},
                                            ParameterVector{input_r});
    }
}
```

Add to the test's `CMakeLists.txt`:
```cmake
add_test_target(ov_transformations_func_tests
    SOURCES test_<pass_name>.cpp
)
```

**Functional / layer tests (when externally observable):**
- If fusion changes user-visible behavior or a performance-sensitive execution path,
  add/extend functional tests under `tests/layer_tests/` or `tests/functional/`.
- Cover representative input shapes/dtypes and backend-relevant conditions.
- Keep tests minimal and deterministic.

### Step 7: Build, Verify and Generate Patch

```bash
# Build only the transformations library and its tests
cmake --build build/ --target ov_transformations_func_tests -j$(nproc)

# Run only this test
./build/src/common/transformations/tests/ov_transformations_func_tests \
  --gtest_filter="*FuseMyOp*"
```

All three tests must pass:
- `FuseMyOp_Basic` → PASSED
- `FuseMyOp_Negative_DynamicWeights` → PASSED
- `FuseMyOp_OutputTypePreserved` → PASSED

**Validate and finalize before generating the patch:**
- Ensure the pass is deterministic and idempotent.
- Verify no duplicated matcher registration.
- Confirm no fallback behavior silently changes model semantics.

#### Generate the patch

```bash
git add src/common/transformations/
git commit -m "transformations: add FuseMyOp MatcherPass

Fuses MatMul(transB=true) + Add(constant_bias) into a single
MyFusedOp to reduce kernel dispatch overhead.

Refs: #<issue_number>"

git format-patch HEAD~1 --stdout > \
  transformation-fuse-my-op-${GITHUB_RUN_ID}.patch
```

The patch is available in `agent-results/transformation/` for collection by the orchestrator.

---

## Fusion Design Recommendations

- Prefer root-cause optimization: fuse only when the resulting graph is strictly safer/faster or equivalent.
- Favor existing OpenVINO ops over introducing new custom ops unless required.
- Keep pattern predicates robust for partial/dynamic shapes.
- Avoid expensive checks in hot transformation loops when cheaper predicates exist.
- Reuse canonical helper utilities for constants, shape checks, and node replacement.
- Prefer `ov::pass::MatcherPass` for local rewrite; use `GraphRewrite` only when grouping multiple related matchers.
- Avoid hidden behavior changes; if assumptions are violated, do not fuse.

---

## Design Checklist (before declaring success)

- [ ] Pattern fires exactly once per matching sub-graph (not multiple times)
- [ ] Pattern does NOT fire on known negative cases (non-constant inputs, wrong consumer count)
- [ ] Friendly name is preserved on the fused node
- [ ] RT info propagated with `copy_runtime_info`
- [ ] Output type and shape inferred correctly after fusion
- [ ] Pass registered at the correct position in the pipeline
- [ ] Both header and source files added to CMakeLists.txt
- [ ] At least 3 unit tests written (positive, negative, type-preservation)
- [ ] Patch generated and saved to agent-results/transformation/

---

## Notes

- Do not claim fusion support without both positive and negative tests.
- Avoid broad refactoring unrelated to the fusion objective.
- Preserve compatibility with existing passes and test baselines.
- Only touches `src/common/transformations/` (general) or the specific plugin path
  if the transformation is backend-exclusive — never both in the same implementation.

---

## Key References

- Transformation API docs: https://docs.openvino.ai/2025/documentation/openvino-extensibility/transformation-api.html
- Existing fusion examples:
  - `src/common/transformations/src/transformations/common_optimizations/fuse_u4_weights_zero_point.cpp`
  - `src/common/transformations/src/transformations/common_optimizations/matmul_multiply_fusion.cpp`
- Test utilities: `src/common/transformations/tests/utils/compare_graphs.hpp`

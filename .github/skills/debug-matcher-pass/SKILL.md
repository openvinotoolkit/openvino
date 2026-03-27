---
name: debug-matcher-pass
description: 'Debug why an OpenVINO MatcherPass is not being applied to a model. Use when a transformation X is expected to fire but does not, when a pattern matcher never triggers, when a pass appears to skip certain nodes, or when investigating "transformation not applied" or "matcher not triggered" issues. Collects matcher debug logs via OV_MATCHER_LOGGING, diagnoses the root cause from the log messages, locates existing unit tests for the transformation, adds a failing reproducer test case, and proposes concrete fix strategies.'
---

# Debug MatcherPass Skill

An end-to-end workflow for diagnosing why an OpenVINO `MatcherPass`-based transformation is not applied to a model. Given a run command and a transformation name, this skill collects matcher logs, identifies the deepest failure point in the matching tree, writes a reproducing unit test, and suggests remediation.

## When to Use This Skill

- User says "transformation X is not applied" or "matcher X never fires"
- A pass is registered in the pipeline but has no visible effect on the graph
- You need to understand at which pattern node the match breaks down
- You want to add a regression/reproducer test for a transformation bug

## Prerequisites

- Build configured with `ENABLE_OPENVINO_DEBUG=ON` (see Step 1 — only needed once)
- The transformation class name or matcher name (e.g., `EliminateSplitConcat`)
- A concrete run command that exercises the failing model/path

---

## Step 1: Verify Debug Build

The matcher logging macros are compiled in only when `ENABLE_OPENVINO_DEBUG=ON`. Check whether the current build already has it:

```bash
grep -i "ENABLE_OPENVINO_DEBUG" build/*/CMakeCache.txt
```

If the flag is absent or set to `OFF`, reconfigure CMake:

```bash
cmake -B build/Debug -DCMAKE_BUILD_TYPE=Debug -DENABLE_OPENVINO_DEBUG=ON <source_dir>
cmake --build build/Debug --parallel
```

> **Note:** Logging also works in `Release` builds as long as `ENABLE_OPENVINO_DEBUG=ON` is set at configure time.

---

## Step 2: Collect Matcher Logs

Follow the instructions in [matcher_logging.md](../../../src/common/transformations/docs/debug_capabilities/matcher_logging.md) to enable `OV_MATCHER_LOGGING` and collect logs. Use `OV_MATCHERS_TO_LOG=<TransformationName>` to filter output to only the transformation of interest. Redirect output to a file for easier inspection.

---

## Step 3: Analyze the Log — Identify Root Cause

Open the log file and search for the transformation name. The log uses a tree structure with `{` (open block) and `}` (close block) markers, and colors to signal success (green) or failure (red).

### 3a. Check if the matcher ran at all

Search the log for the transformation name:

- **Not found at all** → The matcher is either not registered in the pass pipeline for this execution path, or the graph is empty before this pass runs. Check pass registration order (`pass::Manager`).
- **Found** → Proceed to find the failure reason.

### 3b. Find the outermost failure

Search for:

```
END: PATTERN DIDN'T MATCH
```

This confirms the pattern attempted to match but failed. If this line is **never** followed by `END: PATTERN MATCHED`, the transformation never fires.

### 3c. Find the innermost (root) failure

Work inward through the nested blocks to find the deepest `}` labeled with a failure. The following table maps log phrases to root causes:

| Log phrase | Root cause |
|---|---|
| `NODES' TYPE DIDN'T MATCH. EXPECTED: X. OBSERVED: Y` | The op type in the graph does not match the `WrapType` in the pattern. Most commonly an unexpected node has been inserted between two nodes the pattern expects to be directly connected (e.g., a `Convert` or `Reshape` inserted by a prior pass); alternatively, the candidate is simply a different op type. In rare cases the cause is an opset version mismatch (e.g., pattern expects `opset3::ShapeOf`, graph has `opset1::ShapeOf`). |
| `NODES' TYPE MATCHED, but PREDICATE FAILED` | The op type matches but the lambda inside `WrapType(...)` returned `false`. Inspect the predicate in the transformation source — common checks: element type, rank, dynamic shapes, consumer count. |
| `PREDICATE \`name\` FAILED` | A named `pattern::op::Predicate` with this name returned `false`. Locate it by name in the transformation source file. |
| `NUMBER OF ARGUMENTS DOESN'T MATCH. EXPECTED: N. OBSERVED: M` | The graph node has a different number of inputs than the pattern expects. The model graph has a different input structure. |
| `ARGUMENT N DIDN'T MATCH` | The N-th input of the candidate node fails its sub-pattern recursively. |
| `NONE OF OR BRANCHES MATCHED` | A `pattern::op::Or` exhausted all alternative branches. Check each `BRANCH N DIDN'T MATCH` sub-block. |
| `NONE OF PERMUTATIONS MATCHED` | Commutative op (e.g., Add, Multiply): tried all argument orderings but none satisfied the pattern. |
| `LABEL DIDN'T MATCH` | A captured label's value does not satisfy its constraint (e.g., shape symbol equality, consumer requirements). |
| `BLOCK "name" DIDN'T MATCH` | A grouped pattern block failed — look inside the block for the deeper cause. |
| `ATTRIBUTES MISMATCH: VALUE OF \`attr\` IS ... EXPECTED ...` *(verbose only)* | A specific attribute value (axis, group, mode, etc.) does not match the expected value in the pattern. |
| `ANY INPUT DID'T MATCH BECAUSE OF PREDICATE` | A label wrapping `any_input()` has an attached predicate that failed. |

### 3d. Common root causes in OpenVINO transformations

1. **Unexpected intermediate node** — A prior pass inserted an extra node (e.g., `Convert`, `Reshape`, `Transpose`) between two nodes the pattern expects to be directly connected, so the type observed at that edge doesn't match the pattern.
2. **Wrong op type** — The candidate node is simply a different operation than the pattern expects (e.g., `Multiply` vs `MatMul`).
3. **Predicate type/shape check** — `WrapType` lambda checks element type (`f32` vs `f16`), rank, dynamic vs static shapes, or broadcast type. The actual node doesn't qualify.
4. **Consumer count constraint** — Pattern checks that a node has exactly one consumer (`node->output(0).get_target_inputs().size() == 1`), but the graph has multiple consumers.
5. **Pass run order** — A prerequisite transformation (e.g., `ConstantFolding`) has not yet run.
6. **Output index** — Pattern matches `node->output(0)` but the graph provides `node->output(1)`.
7. **Attribute value mismatch** — Pattern constrains an attribute (e.g., `group == 1`, `axis == 0`) that doesn't match the actual node.
8. **Transformation was already applied** — Graph was modified by a symmetric or overlapping pass earlier; the target op no longer exists.
9. **Wrong opset version** — Pattern uses `opset::OpX` but the frontend or a prior pass has already replaced it with a different version or a decomposed form.

---

## Step 4: Locate Unit Tests for the Transformation

Search for existing tests using the transformation class name and/or its header:

```bash
# By class name
grep -rl "TransformationX" src/common/transformations/tests/
grep -rl "TransformationX" src/plugins/*/tests/ tests/layer_tests/

# By header include
grep -rl "transformations/path/to/transformation_x.hpp" \
    src/common/transformations/tests/ \
    src/plugins/*/tests/
```

Common test locations:
- `src/common/transformations/tests/common_optimizations/` — most shared passes
- `src/common/transformations/tests/op_conversions/` — op-conversion passes
- `src/plugins/<plugin>/tests/functional/` — plugin-specific transformations

---

## Step 5: Add a Reproducer Test Case

Once you have identified the failing scenario from the log (Step 3), add a minimal test to the existing test file for the transformation.

### Test structure template

Use `TransformationTestsF` (from `common_test_utils/ov_test_utils.hpp`), which provides `model`, `model_ref`, and `manager` members and automatically compares them after the pass runs:

```cpp
TEST_F(TransformationTestsF, TransformationX_DescribeFailingScenario) {
    {
        // 1. Build the input model that should trigger TransformationX.
        //    Replicate the exact op type / attribute values / graph topology
        //    that appears in the log as the candidate but fails to match.
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});

        // ... construct the ops that match (or almost match) the pattern ...

        model = std::make_shared<Model>(OutputVector{/* output */}, ParameterVector{data});

        // 2. Register the target pass
        manager.register_pass<ov::pass::InitNodeInfo>();
        // add any other prerequisite passes that should run first if needed
        manager.register_pass<ov::pass::TransformationX>();
    }

    {
        // 3. Build the expected reference model after the transformation.
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});

        // ... construct the expected result ops ...

        model_ref = std::make_shared<Model>(OutputVector{/* output */}, ParameterVector{data});
    }
}
```

`TransformationTestsF` runs `manager` on `model` and then compares it against `model_ref` structurally. If the transformation does not fire, the comparison will fail.

### Key tips

- **Mirror the root cause in the test**: if `NODES' TYPE DIDN'T MATCH` showed an unexpected `Convert` inserted between two ops, include that `Convert` in the input model so the test proves the transformation handles (or correctly skips) that topology.
- **Name the test clearly**: include the scenario that was failing (e.g., `EliminateSplitConcat_DifferentSplitAxis`).

---

## Step 6: Propose Resolution Strategies

Based on the root cause identified in Step 3, suggest one or more of the following to the user:

| Root cause | Resolution |
|---|---|
| Unexpected intermediate node | Extend the pattern to optionally absorb the inserted node (e.g., wrap it in `pattern::op::Optional`), or ensure the pass responsible for inserting that node runs after this transformation |
| Wrong op type | Verify the model topology; if the op is a valid alternative, add it to the `WrapType` list |
| Predicate fails (type/shape) | Relax or correct the predicate; run shape/type inference before the pass if shapes are unresolved |
| Consumer count check fails | The single-consumer guard is usually a correctness constraint: replacing a node that feeds multiple consumers would silently alter other paths in the graph. First verify whether skipping is the correct behavior for this model. If the transformation is provably safe for all consumers, remove the consumer-count check from the predicate. If the constraint is intentional, the model simply does not qualify and no fix is needed in the transformation. |
| Pass not registered / wrong order | Register the pass in the correct pipeline position |
| Attribute mismatch | Either extend the pattern to cover the additional attribute values, or verify that the model is correct and not pathological |
| Transformation already applied | Check the pass order — a duplicate or conflicting pass may be consuming the node earlier |
| Output index mismatch | Adjust the pattern to match the correct output index, or add an `Or` branch covering both |
| Wrong opset version in pattern | Update `WrapType` in the transformation to include the correct opset, e.g., `wrap_type<opset1::ShapeOf, opset3::ShapeOf>(predicate)` |

Always pair the resolution suggestion with:
1. The exact file and line in the transformation source where the fix should be made.
2. The test case added in Step 5 as the regression guard.

---

## References

- Matcher logging documentation: [src/common/transformations/docs/debug_capabilities/matcher_logging.md](../../../src/common/transformations/docs/debug_capabilities/matcher_logging.md)
- Log macro definitions: [src/core/dev_api/openvino/core/log_util.hpp](../../../src/core/dev_api/openvino/core/log_util.hpp)
- Matcher implementation: [src/core/src/pattern/matcher.cpp](../../../src/core/src/pattern/matcher.cpp)
- Example transformation tests: [src/common/transformations/tests/common_optimizations/](../../../src/common/transformations/tests/common_optimizations/)

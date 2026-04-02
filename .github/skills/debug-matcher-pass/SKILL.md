---
name: debug-matcher-pass
description: Debug why an OpenVINO MatcherPass transformation is not firing. Use this skill immediately when a user says a transformation is "not applied", a "pass has no effect", a "matcher never triggers", a pattern "doesn't match", a "callback never fires", "WrapType predicate is too strict", a subgraph "not fused" despite the pass being registered, or they see "END: PATTERN DIDN'T MATCH" in matcher logs. Also trigger when a MatcherPass works on one model but silently skips another, when the user wants to add a reproducer test for a transformation that should fire but doesn't, or when they suspect an opset version mismatch preventing a match. Do NOT trigger for: writing a new MatcherPass from scratch, debugging a pass that fires but produces wrong numerical results, crashes in pass registration, or general questions about what MatcherPass is.
---

# Debug MatcherPass Skill

An end-to-end workflow for diagnosing why an OpenVINO `MatcherPass`-based transformation does not fire on a given model.

## Goal

Produce two deliverables before finishing:
1. **Diagnosis report** — post in the chat using the template at the end of this skill. Do not save it to a file.
2. **Reproducer test** — edit the existing test source file to add the new `TEST_F` case, then compile and confirm it exposes the failure.

The only filesystem change should be the test file edit. Do not create additional output files.

---

## Step 0: Gather Prerequisites

Before touching any files or logs, confirm:
- The **transformation class name** (e.g., `EliminateSplitConcat`).
- The **run command** that exercises the failing model/path (benchmark_app invocation, a unit test binary, a Python script, etc.).
- The **build directory** — default to `build/Release` if not specified by the user.

Ask the user only if the transformation name or run command is missing. The build directory is never a blocker.

---

## Step 1: Verify Debug Build

The matcher logging macros are compiled in only when `ENABLE_OPENVINO_DEBUG=ON`. Check whether the current build already has it:

```bash
grep -i "ENABLE_OPENVINO_DEBUG" build/*/CMakeCache.txt
```

If the flag is absent or set to `OFF`, reconfigure CMake using the existing build directory and build type from Step 0:

```bash
cmake -B <build_dir> -DENABLE_OPENVINO_DEBUG=ON
cmake --build <build_dir> --parallel
```

> **Note:** Logging also works in `Release` builds as long as `ENABLE_OPENVINO_DEBUG=ON` is set at configure time.

---

## Step 2: Collect Matcher Logs

Run the failing command with matcher logging enabled and redirect output to a file:

```bash
# Log only the transformation of interest (replace TransformationName):
OV_MATCHER_LOGGING=true \
OV_MATCHERS_TO_LOG=TransformationName \
<your_run_command> 2>&1 | tee matcher.log
```

Additional env vars:
- **`OV_MATCHERS_TO_LOG`** — comma-separated list of matcher names to filter (omit to log all matchers, which produces very large output).
- **`OV_VERBOSE_LOGGING=true`** — prints additional node details (element type, shape, attributes); use when the basic log does not identify the failure clearly.

---

## Step 3: Analyze the Log — Identify Root Cause

Open the log file and search for the transformation name. The log uses a tree structure with `{` (open block) and `}` (close block) markers, and colors to signal success (green) or failure (red).

### 3a. Check if the matcher ran at all

Search the log for the transformation name:

- **Not found at all** → The matcher is either not registered in the pass manager for this execution path, or it runs on an already-transformed/empty graph. Verify registration:

  ```bash
  # Locate where the pass is registered in the pipeline:
  grep -rn "register_pass<.*TransformationName" src/ --include="*.cpp" --include="*.hpp"
  ```

  Check that the pipeline is reachable from your run command (e.g., the correct plugin, the correct compilation path). If the pass is registered, confirm the graph is non-empty before it runs by adding a temporary `ov::pass::VisualizeTree` pass immediately before it.

- **Found** → Proceed to 3b.

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

### 3e. No output at all from OV_MATCHER_LOGGING

If the log file is empty or contains no matcher output even though the flag is set:

- Confirm `ENABLE_OPENVINO_DEBUG=ON` in `CMakeCache.txt` for the binary you are actually running (not a different build directory).
- Confirm the env var is exported in the same shell context as the run command: `export OV_MATCHER_LOGGING=true` or prefix it inline.
- If calling through Python or a launcher script, the env var must survive into the child process — use `os.environ` or prefix the full command.
- Verify you are running the freshly rebuilt binary, not a cached one from a different `bin/` location.

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
    // Do NOT define model_ref here — leave it unset so TransformationTestsF
    // auto-clones the input model as the reference.
}
```

`TransformationTestsF` runs `manager` on `model` and compares it against `model_ref`. When `model_ref` is not explicitly set, it is auto-cloned from the input `model` *before* the pass runs. This means: if the transformation does not fire (the bug is present), `model` is unchanged and equals the clone → **the test passes (green)**. A green test here is the confirmation that the bug is reproduced — the transformation did not alter the graph.

### Key tips

- **Mirror the root cause in the test**: if `NODES' TYPE DIDN'T MATCH` showed an unexpected `Convert` inserted between two ops, include that `Convert` in the input model so the test proves the transformation handles (or correctly skips) that topology.
- **Name the test clearly**: include the scenario that was failing (e.g., `EliminateSplitConcat_DifferentSplitAxis`).

### Build and run the new test

Find the CMake test target that owns the test file:

```bash
grep -rn "add_executable\|ov_add_test_target" \
    src/common/transformations/tests/CMakeLists.txt \
    src/plugins/*/tests/CMakeLists.txt | grep -i transformation
```

Build only that target and run the new test:

```bash
cmake --build <build_dir> --target <test_binary_name> --parallel

bin/intel64/<build_type>/<test_binary_name> --gtest_filter="*TransformationName_DescribeFailingScenario*"
```

**Expected outcome before the fix:** the test passes (green). This is correct — the transformation did not fire, so the model is unchanged and matches the auto-cloned `model_ref`. A passing test here is a *necessary* but not *sufficient* confirmation that the bug is reproduced.

### Validate the reproducer with matcher logging

A green test alone is not enough: the test could be passing for the wrong reason (e.g., the transformation was never even attempted on the small test graph, rather than failing at the expected point). Always cross-check by re-running the test binary with matcher logging enabled and verifying that the log shows the **same failure phrase and the same node** as in the original model log from Step 3:

```bash
OV_MATCHER_LOGGING=true \
OV_MATCHERS_TO_LOG=TransformationName \
bin/intel64/<build_type>/<test_binary_name> \
    --gtest_filter="*TransformationName_DescribeFailingScenario*" 2>&1 | tee matcher_test.log
```

Compare `matcher_test.log` against the original `matcher.log`:
- The same innermost failure phrase (e.g., `NODES' TYPE DIDN'T MATCH. EXPECTED: X. OBSERVED: Y`) must appear.
- The failing node type must match.

If the logs diverge (e.g., the test log shows a different failure phrase, or `END: PATTERN DIDN'T MATCH` never appears at all), the test model does not faithfully reproduce the original bug — revise the test graph to closer match the topology identified in Step 3 and repeat.

Once both conditions are met (test is green **and** logs agree), the reproducer is valid. Record the relevant log excerpt in the diagnosis report.

Once the fix is implemented and the transformation fires correctly, the test will fail because `model` no longer matches the unmodified clone. At that point, add an explicit `model_ref` block with the expected transformed graph to turn it into a proper regression guard.

---

## Step 6: Propose Resolution Strategies

Based on the root cause identified in Step 3, suggest one or more of the following to the user:

| Root cause | Resolution |
|---|---|
| Unexpected intermediate node | Extend the pattern to optionally absorb the inserted node (e.g., wrap it in `pattern::op::Optional`), or ensure the pass responsible for inserting that node runs after this transformation |
| Wrong op type | Verify the model topology; if the op is a valid alternative, add it to the `WrapType` list |
| Predicate fails (type/shape) | Relax or correct the predicate; run shape/type inference before the pass if shapes are unresolved |
| Consumer count check fails | The single-consumer guard is usually a correctness constraint: replacing a node that feeds multiple consumers would silently alter other paths in the graph. First verify whether skipping is the correct behavior for this model. If the transformation is provably safe for all consumers, remove the consumer-count check from the predicate. If the constraint is intentional, the model simply does not qualify and no fix is needed in the transformation. |
| Pass not registered / wrong order | Register the pass in the correct pipeline position; use `grep -rn "register_pass"` to find the pipeline source |
| Attribute mismatch | Either extend the pattern to cover the additional attribute values, or verify that the model is correct and not pathological |
| Transformation already applied | Check the pass order — a duplicate or conflicting pass may be consuming the node earlier |
| Output index mismatch | Adjust the pattern to match the correct output index, or add an `Or` branch covering both |
| Wrong opset version in pattern | Update `WrapType` in the transformation to include the correct opset, e.g., `wrap_type<opset1::ShapeOf, opset3::ShapeOf>(predicate)` |

Always pair the resolution suggestion with:
1. The exact file and line in the transformation source where the fix should be made.
2. The test case added in Step 5 as the regression guard.

### When to stop and escalate

If after Steps 1–3 you find:
- The target op is provably absent from the graph (confirmed via `VisualizeTree` before the pass) and no prior pass should have produced it → the problem is upstream of the matcher (wrong frontend conversion, wrong pass pipeline entry point). Diagnose the upstream issue separately.
- The pass is registered, the matcher runs, but a structurally identical sub-graph is silently skipped alongside one that correctly fires → the model may have an intentional correctness guard (e.g., consumer count, const-foldability). Verify this is intentional before proposing a change.

---

## Diagnosis Report Template

When both deliverables are complete, post the following in the chat (do not save to a file).
See [references/example-diagnosis-report.md](references/example-diagnosis-report.md) for a fully filled example — use it as a quality bar for level of detail, especially for *Log evidence* (must be a direct quote from the log, not paraphrased from source) and *Resolution* (must include file + line).

```
## MatcherPass Diagnosis: <TransformationName>

**Root cause:** <one-sentence summary>
**Log evidence:** `<exact log phrase that identified the failure>`
**Failing node:** <op type, location in graph>
**Resolution:** <what needs to change and where (file:line)>

## Reproducer Test
File: <path to test file>
Test name: `<TEST_F name>`
Status before fix: FAIL (confirms reproduction)
```

---

## References

- **Filled example report:** [references/example-diagnosis-report.md](references/example-diagnosis-report.md)
- Matcher logging documentation: [src/common/transformations/docs/debug_capabilities/matcher_logging.md](../../../src/common/transformations/docs/debug_capabilities/matcher_logging.md)
- Log macro definitions: [src/core/dev_api/openvino/core/log_util.hpp](../../../src/core/dev_api/openvino/core/log_util.hpp)
- Matcher implementation: [src/core/src/pattern/matcher.cpp](../../../src/core/src/pattern/matcher.cpp)
- Example transformation tests: [src/common/transformations/tests/common_optimizations/](../../../src/common/transformations/tests/common_optimizations/)

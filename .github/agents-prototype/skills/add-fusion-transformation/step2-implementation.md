# Skill: OpenVINO Transformation Implementation

## Purpose

Implement a single OpenVINO `MatcherPass` / `FunctionPass` / `BackwardGraphRewrite`
from the `transformation_analysis.md` produced by the analysis skill.
Covers: header, source, CMake registration, pass-pipeline registration.

## When to invoke

- Transformation Agent Step 2 (after analysis is complete)
- When a `patch_type=transformation` git patch exists and needs refinement

---

## Steps

### Step 1: Create Header File

Location: `src/common/transformations/include/transformations/<domain>/<pass_name>.hpp`

Read an existing transformation header as your template. Good references:
- `src/common/transformations/include/transformations/common_optimizations/fuse_u4_weights_zero_point.hpp`
- `src/common/transformations/include/transformations/common_optimizations/matmul_multiply_fusion.hpp`

Every header must include:
- `OPENVINO_RTTI("ClassName", "0")` macro
- `TRANSFORMATIONS_API` export macro
- A doxygen comment with before/after ASCII diagram

Do not copy-paste the template file directly; read and adapt the structure.

### Step 2: Create Source File

Location: `src/common/transformations/src/<domain>/<pass_name>.cpp`

Use the template below as your starting point, then read the closest existing transformation to
adapt the pattern to your sub-graph:

```cpp
// Starting template — adapt node types, input count, and guards to your sub-graph.

<PassName>::<PassName>() {
    MATCHER_SCOPE(<PassName>);

    // Describe the pattern bottom-up (root last):
    auto input   = pattern::any_input();
    auto weights = pattern::wrap_type<ov::op::v0::Constant>();
    auto root_op = pattern::wrap_type<ov::op::vN::SomeOp>({input, weights});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pmap = m.get_pattern_map();
        const auto& root = pmap.at(root_op);

        // Guard: skip if node is shared (multiple consumers)
        if (root->get_output_target_inputs(0).size() != 1)
            return false;

        auto fused = std::make_shared<ov::op::vN::FusedOp>(
            pmap.at(input), pmap.at(weights));
        ov::copy_runtime_info({root}, fused);
        fused->set_friendly_name(root->get_friendly_name());
        ov::replace_node(root, fused);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root_op, matcher_name);
    register_matcher(m, callback);
}
```

Key implementation patterns to follow from the template:

- Use `MATCHER_SCOPE(ClassName)` at the start of the constructor
- Use `pattern::wrap_type<Op>({...})` for node matching
- Use `pattern::consumers_count(N)` to guard on consumer count
- Call `ov::copy_runtime_info({old_nodes...}, new_node)` before replacement
- Call `fused->set_friendly_name(root->get_friendly_name())`
- Call `ov::replace_node(old_root, new_node)` as the last step
- Return `true` from callback on successful replacement

Real transformations to read before adapting your code:
- `src/common/transformations/src/transformations/common_optimizations/fuse_u4_weights_zero_point.cpp`
- `src/common/transformations/src/transformations/common_optimizations/matmul_multiply_fusion.cpp`

For `FunctionPass` (full-graph traversal), read:
- `src/common/transformations/src/transformations/common_optimizations/align_mixed_fp32_fp16_types.cpp`

### Step 3: Update CMakeLists.txt

The CMake build system picks up any `.cpp` file listed in the `LIBRARY_SRC` variable.
Add your new `.cpp` to `src/common/transformations/CMakeLists.txt` in the `LIBRARY_SRC` set.
Add the corresponding `.hpp` to `PUBLIC_HEADERS`.

**Do not add a new `add_library` or `add_test_target` call** — the existing targets
already compile everything in `LIBRARY_SRC`.

### Step 4: Register in the Pass Pipeline

**For general (all-backend) use:**
```cpp
// src/common/transformations/src/transformations/common_optimizations/common_optimizations.cpp
// Find the block of ADD_MATCHER calls. Insert in priority order.
ADD_MATCHER(manager, FuseMyOp)
```

**For backend-specific use (CPU only):**
```cpp
// src/plugins/intel_cpu/src/graph_optimizer.cpp
ApplyCommonOptimizations::run_on_model(model) {
    ...
    manager.register_pass<ov::pass::FuseMyOp>();
    ...
}
```

**With pass config (opt-in by consumer):**
```cpp
manager.register_pass<ov::pass::FuseMyOp>();
// Disabled by default; backend enables explicitly:
pass_config->enable<ov::pass::FuseMyOp>();
```

### Step 5: Self-Check

After writing implementation and tests, verify the transformation fires correctly.
Run the cross-platform validation script:

```
python .github/scripts/meat/validate_transformation.py --pass-name <ClassName>
```

The script loads the model at `agent-results/analyze-and-convert/ov_model_*/openvino_model.xml`,
applies the transformation, and reports whether the fused op appears in the resulting graph.

---

## Common Pitfalls

| Problem | Fix |
|---|---|
| Pass fires but output shape is wrong | Call `fused->validate_and_infer_types()` after replacing |
| Pattern matches but replacement crashes | Ensure all `shared_ptr` casts are checked with `ov::as_type_ptr` |
| Pass fires on non-constant input | Add `pattern::wrap_type<Constant>()` with a predicate check in callback |
| Pass does not fire at all | Verify node consumer count (`consumers_count`) and output port indices |
| Accuracy drops after fusion | Check that the fused op handles transposed inputs the same way as the original sub-graph |

---

## Output

Modified files:
- `src/common/transformations/include/transformations/<domain>/<pass_name>.hpp`
- `src/common/transformations/src/<domain>/<pass_name>.cpp`
- `src/common/transformations/CMakeLists.txt`
- `src/common/transformations/src/transformations/common_optimizations/common_optimizations.cpp`
  (or backend-specific file)

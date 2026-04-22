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

Location:
```
src/common/transformations/include/transformations/<domain>/<pass_name>.hpp
```

Minimum header template:
```cpp
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// @brief Fuses <describe pattern> into a single <TargetOp>.
///
/// Before:
///   MatMul + Add(bias) → fuse into LinearOp
///
/// After:
///   LinearOp(input, weights, bias)
class TRANSFORMATIONS_API FuseMyOp : public MatcherPass {
public:
    OPENVINO_RTTI("FuseMyOp", "0");
    FuseMyOp();
};

}  // namespace ov::pass
```

For `FunctionPass`:
```cpp
class TRANSFORMATIONS_API MyFunctionPass : public FunctionPass {
public:
    OPENVINO_RTTI("MyFunctionPass", "0");
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};
```

### Step 2: Create Source File

Location:
```
src/common/transformations/src/<domain>/<pass_name>.cpp
```

#### `MatcherPass` body (most common case)
```cpp
#include "transformations/<domain>/<pass_name>.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

// Include the new fused op header (from Core OpSpec outputs)
#include "ov_ops/my_fused_op.hpp"

ov::pass::FuseMyOp::FuseMyOp() {
    MATCHER_SCOPE(FuseMyOp);

    // --- Pattern definition ---
    auto input   = pattern::any_input(pattern::has_static_rank());
    auto weights = pattern::wrap_type<ov::op::v0::Constant>();
    auto matmul  = pattern::wrap_type<ov::op::v0::MatMul>({input, weights});
    auto bias    = pattern::wrap_type<ov::op::v0::Constant>();
    auto add     = pattern::wrap_type<ov::op::v1::Add>(
                       {matmul, bias},
                       pattern::consumers_count(1));  // add must have exactly one consumer

    // --- Callback ---
    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pmap   = m.get_pattern_map();
        const auto& matmul_node = pmap.at(matmul);
        const auto& bias_node   = std::dynamic_pointer_cast<ov::op::v0::Constant>(pmap.at(bias));
        const auto& add_node    = pmap.at(add);

        // Build fused op
        auto fused = std::make_shared<ov::op::internal::MyFusedOp>(
            matmul_node->input_value(0),   // input
            matmul_node->input_value(1),   // weights
            bias_node);                    // bias

        // Preserve name and RT info
        fused->set_friendly_name(add_node->get_friendly_name());
        ov::copy_runtime_info({matmul_node, add_node}, fused);

        // Replace
        ov::replace_node(add_node, fused);
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(add, matcher_name);
    register_matcher(m, callback);
}
```

Key helpers:
- `pattern::consumers_count(N)` — gate on number of output consumers
- `pattern::has_static_rank()` — only match when rank is known at compile time
- `pattern::type_matches(element::f32)` — only match specific dtype
- `ov::copy_runtime_info({...}, new_node)` — propagate runtime metadata
- `ov::replace_node(old, new)` — the only safe node replacement method

#### `FunctionPass` body
```cpp
bool ov::pass::MyFunctionPass::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    for (const auto& node : model->get_ordered_ops()) {
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
            // Check conditions, build replacement, call replace_node
            changed = true;
        }
    }
    return changed;
}
```

### Step 3: Update CMakeLists.txt

In `src/common/transformations/CMakeLists.txt`, add the new `.cpp` source:
```cmake
set(LIBRARY_SRC
    ...
    src/<domain>/<pass_name>.cpp
    ...
)

set(PUBLIC_HEADERS
    ...
    include/transformations/<domain>/<pass_name>.hpp
    ...
)
```

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

Run a quick Python validation before writing tests:
```python
import openvino as ov
import openvino.transformations as ovt

model = ov.Core().read_model("openvino_model.xml")
manager = ov.pass.Manager()
manager.register_pass(ovt.FuseMyOp())
manager.run_passes(model)

# Check the fused op is present
fused_ops = [n for n in model.get_ordered_ops()
             if n.get_type_name() == "MyFusedOp"]
assert len(fused_ops) > 0, "Transformation did not fire!"
print(f"Fused {len(fused_ops)} op(s) successfully")
```

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

# Skill: Add a New Operation Translator to the OpenVINO ONNX Frontend

---

## 1. Prerequisites

Before writing any code:

- [ ] Read the [ONNX spec for the op](https://onnx.ai/onnx/operators/) — inputs, outputs, attributes, type constraints, and behavioral notes.
- [ ] Check the [ONNX Operator Changelog](https://github.com/onnx/onnx/blob/main/docs/Changelog.md) for all opset versions that introduced or modified the op.
- [ ] Identify the matching OpenVINO op(s) in `src/core/include/openvino/op/`. If none exists, use a composition of existing ops or escalate to the Core OpSpec Agent.
- [ ] Search `src/frontends/common_translators/` — a shared translator may already exist or be partially implemented.
- [ ] Check if other frontends (PyTorch, TF) already have a translator for the same operation; it may be reusable or a model to follow.

---

## 2. Key Source Locations

| Path | Description |
|---|---|
| `src/frontends/onnx/frontend/src/op/` | Per-op translator files — create your new file here. |
| `src/frontends/onnx/frontend/src/ops_bridge.cpp` | Operator registry — add `ONNX_OP` registration here. |
| `src/frontends/onnx/frontend/src/core/operator_set.hpp` | `ONNX_OP` and `ONNX_OP_M` macros. |
| `src/frontends/onnx/frontend/src/version_range.hpp` | `OPSET_SINCE`, `OPSET_RANGE`, `OPSET_IN` helpers. |
| `src/frontends/onnx/frontend/src/core/node.hpp` | `ov::frontend::onnx::Node` — the context object passed to each translator. |
| `src/frontends/onnx/frontend/src/utils/common.hpp` | Shared helpers: `is_input_valid`, `handle_opset6_binary_op`. **Always check here.** |
| `src/frontends/onnx/frontend/CMakeLists.txt` | Add new `.cpp` file here under `SOURCES`. |
| `src/frontends/onnx/tests/onnx_import.in.cpp` | Main import/inference tests. Add test cases here. |
| `src/frontends/onnx/tests/models/` | `.prototxt` test model definitions. Add test models here. |

---

## 3. `ov::frontend::onnx::Node` API Reference

`Node` is the context object every translator receives.

```cpp
#include "core/node.hpp"

// Get all op inputs as OpenVINO outputs:
ov::OutputVector inputs = node.get_ov_inputs();

// Get a specific input:
auto x = node.get_ov_inputs()[0];

// Get an attribute with a default value:
int64_t axis = node.get_attribute_value<int64_t>("axis", 0);
std::vector<int64_t> pads = node.get_attribute_value<std::vector<int64_t>>("pads", {0, 0});
std::string auto_pad = node.get_attribute_value<std::string>("auto_pad", "NOTSET");

// Check if an attribute exists:
bool has_pads = node.has_attribute("pads");

// Check if an optional input is present (not "missing"):
#include "utils/common.hpp"
bool valid = common::is_input_valid(node, 2);
```

---

## 4. Registration Macros

In `ops_bridge.cpp`:

```cpp
#include "op/<new_op>.hpp"

namespace ai_onnx {
namespace opset_1 {
// Define translator in .hpp/.cpp file
}  // namespace opset_1
}  // namespace ai_onnx

// Register with a single opset version range:
ONNX_OP("GridSample", OPSET_SINCE(16), ai_onnx::opset_16::grid_sample);

// Register with version ranges — when behavior changed at opset 11:
ONNX_OP("Resize", OPSET_RANGE(10, 10), ai_onnx::opset_10::resize);
ONNX_OP("Resize", OPSET_RANGE(11, 17), ai_onnx::opset_11::resize);
ONNX_OP("Resize", OPSET_SINCE(18), ai_onnx::opset_18::resize);

// Register for specific opset versions (when there are gaps):
ONNX_OP("Cast", OPSET_IN({1, 6, 9, 13, 19}), ai_onnx::opset_13::cast);
```

---

## 5. Step-by-Step Implementation

### 5a. Create the translator file

```bash
# Create the header file:
cat > src/frontends/onnx/frontend/src/op/<new_op>.hpp << 'EOF'
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {

namespace opset_<first_version> {
ov::OutputVector <new_op>(const ov::frontend::onnx::Node& node);
}  // namespace opset_<first_version>

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
EOF

# Create the source file:
cat > src/frontends/onnx/frontend/src/op/<new_op>.cpp << 'EOF'
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op/<new_op>.hpp"

#include "core/node.hpp"
#include "openvino/op/...hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {

namespace opset_<first_version> {

ov::OutputVector <new_op>(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto x = inputs[0];

    // ... read attributes ...
    const auto axis = node.get_attribute_value<int64_t>("axis", 0);

    // ... build OpenVINO subgraph ...
    return {std::make_shared<ov::op::v0::...>(x)};
}

}  // namespace opset_<first_version>

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
EOF
```

### 5b. Handling multiple opset versions

When an op was added in opset 9 and modified in opset 13:

```cpp
// ---- opset_9 namespace ----
namespace opset_9 {
ov::OutputVector my_op(const ov::frontend::onnx::Node& node) {
    // opset 9 implementation
}
}  // namespace opset_9

// ---- opset_13 namespace ----
namespace opset_13 {
ov::OutputVector my_op(const ov::frontend::onnx::Node& node) {
    // opset 13 implementation (new behavior/attributes)
}
}  // namespace opset_13
```

In `ops_bridge.cpp`:
```cpp
ONNX_OP("MyOp", OPSET_RANGE(9, 12), ai_onnx::opset_9::my_op);
ONNX_OP("MyOp", OPSET_SINCE(13), ai_onnx::opset_13::my_op);
```

### 5c. Handling simple binary/elementwise ops

```cpp
#include "utils/common.hpp"

ov::OutputVector add(const ov::frontend::onnx::Node& node) {
    return common::handle_opset6_binary_op<ov::op::v1::Add>(node);
}
```

### 5d. Using shared helpers

```cpp
#include "utils/common.hpp"

// Check optional input before accessing it:
const auto inputs = node.get_ov_inputs();
if (common::is_input_valid(node, 1)) {
    auto bias = inputs[1];
    // ... use bias ...
}
```

### 5e. Checking and using `common_translators`

```bash
grep -rn '<op_name>' src/frontends/common_translators/include/common_translators.hpp
```

If found, use it directly:

```cpp
#include "common_translators.hpp"

ov::OutputVector my_op(const ov::frontend::onnx::Node& node) {
    return common_translators::translate_<op>(node.get_ov_inputs(), attributes...);
}
```

### 5f. Mark operations for special types

For ops that process complex or quantized tensors:

```cpp
#include "openvino/frontend/complex_type_mark.hpp"

// In normalize-step MatcherPass (if needed):
auto complex_mark = std::make_shared<ComplexTypeMark>(input, input_et);
```

### 5g. Normalize-step transformations

For ops that require post-translation graph rewrites (e.g., resolving Mark operations or merging multi-op patterns):

1. Create a `MatcherPass` in `src/frontends/onnx/frontend/src/transforms/` (or `src/frontends/common_translators/` for shared transforms).
2. Register in `FrontEnd::normalize()` in `frontend.cpp`.

### 5h. Update `CMakeLists.txt`

In `src/frontends/onnx/frontend/CMakeLists.txt`:
```cmake
set(SOURCES
    ...
    src/op/<new_op>.cpp
    ...
)
```

### 5i. Register in `ops_bridge.cpp`

In `src/frontends/onnx/frontend/src/ops_bridge.cpp`:
```cpp
#include "op/<new_op>.hpp"
...
ONNX_OP("<OpName>", OPSET_SINCE(<first_opset>), ai_onnx::opset_<first_opset>::<new_op>);
```

---

## 6. Adding Tests

### 6a. Create a `.prototxt` test model

In `src/frontends/onnx/tests/models/<op_name>.prototxt`:

```protobuf
ir_version: 7
graph {
  node {
    op_type: "GridSample"
    attribute {
      name: "mode"
      s: "bilinear"
      type: STRING
    }
    input: "X"
    input: "grid"
    output: "Y"
  }
  name: "test_grid_sample"
  input {
    name: "X"
    type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } ... }}}
  }
  output {
    name: "Y"
    type { tensor_type { elem_type: 1 shape { ... }}}
  }
}
opset_import { version: 16 }
```

### 6b. Add C++ test cases

In `src/frontends/onnx/tests/onnx_import.in.cpp`:

```cpp
NGRAPH_TEST(${BACKEND_NAME}, onnx_model_<op_name>) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(CommonTestUtils::getExecutionDirectory(),
                             SERIALIZED_ZOO, "onnx/<op_name>.prototxt"));

    auto test_case = test::TestCase(function, s_device);
    test_case.add_input<float>({...});  // input values
    test_case.add_expected_output<float>(Shape{...}, {...});  // expected output
    test_case.run();
}
```

### 6c. Add multiple test scenarios

Cover all important cases:
- Basic case with default attributes
- Non-default attribute values  
- Edge cases (empty tensor, scalar, dynamic rank)
- Multiple opset versions if behavior differs

---

## 7. Build and Test

```bash
# Build the ONNX frontend:
cmake --build build --target openvino_onnx_frontend -j$(nproc)

# Build and run tests:
cmake --build build --target ov_onnx_frontend_tests -j$(nproc)
cd build && ctest -R "ov_onnx_frontend_tests" --output-on-failure -j$(nproc)

# Run a specific test:
ctest -R "onnx_model_<op_name>" -V

# Apply clang-format:
cmake --build build --target clang_format_fix_all -j$(nproc)
```

---

## 8. Validation Checklist

- [ ] **ONNX spec consulted** — all attributes, input/output optionality, type constraints covered.
- [ ] **All opset versions covered** — `ONNX_OP` registrations span the correct version ranges.
- [ ] **Optional inputs guarded** — `common::is_input_valid(node, N)` used for all optional inputs.
- [ ] **Default attribute values** — per ONNX spec for each opset version.
- [ ] **CMakeLists.txt updated** — new `.cpp` added to `SOURCES`.
- [ ] **`ops_bridge.cpp` updated** — `ONNX_OP` registration added.
- [ ] **Test model added** — `.prototxt` covers basic case and edge cases.
- [ ] **C++ test added** — at least one test in `onnx_import.in.cpp`.
- [ ] **All ONNX tests pass** — `ctest -R ov_onnx_frontend_tests` green.
- [ ] **clang-format applied** — no formatting diffs.
- [ ] **Supported ops doc updated** — `src/frontends/onnx/docs/supported_ops.md` (if maintained).

---

## 9. Common Pitfalls

| Pitfall | Explanation |
|---|---|
| **Wrong opset range** | `OPSET_SINCE(11)` instead of `OPSET_RANGE(11, 17)` when the op changed at opset 18. |
| **Attribute type mismatch** | `get_attribute_value<int64_t>()` vs `get_attribute_value<float>()` — wrong type silently returns a default value. |
| **Not checking common_translators** | Many ops are already in `src/frontends/common_translators/`. Check before writing a new one. |
| **Forgetting `is_input_valid()`** | Accessing `inputs[2]` when the input is optional and absent causes a crash. |
| **Missing `#include`** | Each OpenVINO op class needs its header (e.g., `#include "openvino/op/add.hpp"`). |
| **Broadcasting edge cases** | ONNX uses Numpy-style broadcasting; verify with scalar and 0-rank tensor inputs. |
| **Not handling empty attribute list** | Some attributes may have list type with empty default; `get_attribute_value<std::vector<int64_t>>("axis", {})`. |
| **Mark operations missing** | For complex/quantized/sequence types, the normalize-step won't resolve correctly without the corresponding Mark operation. |

---

## 10. Reference Links

- [ONNX Operator Specifications](https://onnx.ai/onnx/operators/)
- [ONNX Operator Changelog](https://github.com/onnx/onnx/blob/main/docs/Changelog.md)
- [OpenVINO Available Operations](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html)
- [OpenVINO ONNX Frontend README](src/frontends/onnx/README.md)
- [OpenVINO ONNX Supported Ops](src/frontends/onnx/docs/supported_ops.md)

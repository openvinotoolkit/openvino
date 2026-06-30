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
| `src/frontends/onnx/frontend/src/op/` | Per-op translator files — create your new `.cpp` file here. `ONNX_OP` registration goes at the bottom of this file. |
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

The `ONNX_OP` macro is placed **at the bottom of the per-op `.cpp` file** (not in `ops_bridge.cpp`). Look at existing op files such as `src/frontends/onnx/frontend/src/op/sequence_insert.cpp` or `src/frontends/onnx/frontend/src/op/resize.cpp` as reference.

**Example** (adapt op name and version range to your op):

```
// Single version range:
ONNX_OP("GridSample", OPSET_SINCE(16), ai_onnx::opset_16::grid_sample);

// Multiple ranges when behavior changed between opsets:
ONNX_OP("Resize", OPSET_RANGE(10, 10), ai_onnx::opset_10::resize);
ONNX_OP("Resize", OPSET_RANGE(11, 17), ai_onnx::opset_11::resize);
ONNX_OP("Resize", OPSET_SINCE(18), ai_onnx::opset_18::resize);

// Specific individual opset versions (when there are gaps):
ONNX_OP("Cast", OPSET_IN({1, 6, 9, 13, 19}), ai_onnx::opset_13::cast);
```

---

## 5. Step-by-Step Implementation

### 5a. Create the translator file

ONNX op translators consist of **a single `.cpp` file only** — no corresponding `.hpp` is needed.

Read an existing op as your template. Good references:
- `src/frontends/onnx/frontend/src/op/sequence_insert.cpp` (simple, recent pattern)
- `src/frontends/onnx/frontend/src/op/resize.cpp` (multi-opset-version example)

Adapt the chosen template to your op. Do not copy-paste blindly — understand each section before writing.

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

```cpp
ONNX_OP("MyOp", OPSET_RANGE(9, 12), ai_onnx::opset_9::my_op);
ONNX_OP("MyOp", OPSET_SINCE(13), ai_onnx::opset_13::my_op);

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
- [ ] **`ONNX_OP` registration added** — macro at bottom of the new `.cpp` translator file (not in `ops_bridge.cpp`).
- [ ] **Test model added** — `.prototxt` covers basic case and edge cases.
- [ ] **C++ test added** — at least one test in `onnx_import.in.cpp`.
- [ ] **Tests pass** — build `ov_onnx_frontend_tests`, run the full suite, confirm the new test is present and green with no regressions. Do not report success without running it.
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
| **Not checking for a closely related existing translator** | If the new op is a quantized/extended variant of an existing op (e.g., `DynamicQuantizeLSTM` vs `LSTM`), read that translator first. Reuse shared utilities (`recurrent.hpp` for RNN-family, etc.) rather than duplicating helper code. |
| **Speculative input handling** | Only handle input ranks/layouts defined by the spec. Adding speculative rank-N handling for cases the spec doesn't define creates dead code that may silently produce wrong outputs (wrong rank on outputs, missing bidirectional guard, etc.). When in doubt, `CHECK_VALID_NODE` to reject unsupported input shapes. |
| **Silently dropping unsupported optional inputs** | If an optional input (e.g., peephole weights) cannot be represented in the OV subgraph, reject it with an explicit `CHECK_VALID_NODE` error rather than ignoring it. Silent drops produce wrong results. |
| **Hand-rolling dequantization subgraphs** | For ops with quantized inputs (int8/uint8 + scale + zero_point), use `ov::decomposition::low_precision_dequantize` (`#include "openvino/decompositions/low_precision_dequantize.hpp"`) instead of manually building Convert→Subtract→Multiply. The helper produces the canonical pattern recognised by `ov::pass::MarkDequantization` and protects the Convert from constant folding, enabling downstream LPT and weight-decompression optimisations. **Scale axis alignment:** `low_precision_dequantize` uses numpy right-aligned autobroadcast, so the scale/zero_point quantization axes must be the trailing dimensions of the weight tensor. If they are not, align them by appending size-1 dimensions: when the scale is a `Constant` (the normal case in quantized models), create a new `Constant` with the reshaped shape via `Constant(const Constant&, Shape)` — no graph node is inserted and `MarkDequantization` still fires. When the scale is a runtime input (e.g. in tests), fall back to an `Unsqueeze` node — LPT will not fire but correctness is preserved. Add both a runtime-input test and a graph-constant test to cover both code paths. |

---

## 10. Reference Links

- [ONNX Operator Specifications](https://onnx.ai/onnx/operators/)
- [ONNX Operator Changelog](https://github.com/onnx/onnx/blob/main/docs/Changelog.md)
- [OpenVINO Available Operations](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html)
- [OpenVINO ONNX Frontend README](src/frontends/onnx/README.md)
- [OpenVINO ONNX Supported Ops](src/frontends/onnx/docs/supported_ops.md)

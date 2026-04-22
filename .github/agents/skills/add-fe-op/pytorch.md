# Skill: Add a New Operation Translator to the OpenVINO PyTorch Frontend

---

## 1. Prerequisites

Before writing any code:

- [ ] Read the [PyTorch documentation](https://pytorch.org/docs/stable/) for the op — function signature, parameter types, and behavioral notes.
- [ ] Check the [PyTorch ATen operator schema](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml) for all overloads.
- [ ] Identify whether the op needs to be supported in **TorchScript** (`aten::op`), **torch.export/FX** (`aten.op.overload`), or both.
- [ ] Identify the matching OpenVINO op(s) in `src/core/include/openvino/op/`. If none exists, escalate to the Core OpSpec Agent.
- [ ] Search `src/frontends/common_translators/` — a shared implementation may already exist.
- [ ] Check if the op can be expressed using existing `translate_1to1_match_*` wrappers.

---

## 2. Key Source Locations

| Path | Description |
|---|---|
| `src/frontends/pytorch/src/op/` | Per-op translator files — create your new file here. |
| `src/frontends/pytorch/src/op_table.cpp` | Operator registry — `get_supported_ops_ts()` and `get_supported_ops_fx()`. Add registration here. |
| `src/frontends/pytorch/include/openvino/frontend/pytorch/node_context.hpp` | `NodeContext` — the context object each translator receives. |
| `src/frontends/pytorch/src/utils.hpp` | Shared helpers: `num_inputs_check`, `make_optional_bias`, `get_shape_rank`, `normalize_axis`, `numel`, `convert_dtype`, `concat_list_construct`. **Always check here first.** |
| `src/frontends/pytorch/src/pt_framework_node.hpp` | `PtFrameworkNode` — placeholder for unconverted ops. |
| `src/frontends/pytorch/src/transforms/` | Normalize-step transformations — `MatcherPass` subclasses that run after initial translation. |
| `src/frontends/pytorch/CMakeLists.txt` | Add new `.cpp` files here. |
| `tests/layer_tests/pytorch_tests/` | Python layer tests — create `test_<op>.py` here. |
| `tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py` | `PytorchLayerTest` base class. |

---

## 3. `NodeContext` API Reference

`NodeContext` is the context object every translator receives.

```cpp
#include "openvino/frontend/pytorch/node_context.hpp"

// Get an input by index:
ov::Output<ov::Node> x = context.get_input(0);

// Get a constant input (already a tensor, reads the value):
int64_t dim = context.const_input<int64_t>(1);
double alpha = context.const_input<double>(2);
std::vector<int64_t> shape = context.const_input<std::vector<int64_t>>(3);

// Check if an optional input is None:
bool is_none = context.input_is_none(2);

// Mark a created node (MANDATORY for every created OV node):
auto add = context.mark_node(std::make_shared<ov::op::v1::Add>(x, y));

// Get the number of inputs:
size_t n = context.get_input_size();
```

---

## 4. Registration in `op_table.cpp`

### Define converter function

```cpp
// In the OP_CONVERTER macro (or standalone function):
OP_CONVERTER(translate_<new_op>) {
    // context is NodeContext&
    num_inputs_check(context, 1, 3);
    const auto x = context.get_input(0);
    // ...
    return {context.mark_node(std::make_shared<ov::op::v0::Relu>(x))};
}
```

### Register in `get_supported_ops_ts()` (TorchScript)

```cpp
{"aten::<op_name>",          ov::frontend::pytorch::op::translate_<new_op>},
```

### Register in `get_supported_ops_fx()` (torch.export / FX)

```cpp
{"aten.<op_name>.default",   ov::frontend::pytorch::op::translate_<new_op>},
{"aten.<op_name>.Tensor",    ov::frontend::pytorch::op::translate_<new_op>},
```

### 1:1 mapping wrappers (no custom logic needed)

```cpp
// Single input → single output, same type:
{"aten::relu",  op::translate_1to1_match_1_inputs<opset10::Relu>},
{"aten.relu.default",  op::translate_1to1_match_1_inputs<opset10::Relu>},

// Two inputs, type-aligned:
{"aten::add.Tensor", op::translate_1to1_match_2_inputs_align_types<opset1::Add>},

// Inplace variant:
{"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
```

---

## 5. Step-by-Step Implementation

### 5a. Create the translator file

```bash
# Create the source file:
cat > src/frontends/pytorch/src/op/<new_op>.cpp << 'EOF'
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/...hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_<new_op>(const NodeContext& context) {
    num_inputs_check(context, 1, 3);  // min, max inputs
    const auto x = context.get_input(0);

    // Handle optional inputs:
    if (!context.input_is_none(1)) {
        const auto bias = context.get_input(1);
        auto add_result = context.mark_node(std::make_shared<v1::Add>(x, bias));
        return {add_result};
    }
    return {context.mark_node(std::make_shared<v0::Relu>(x))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
EOF
```

Also declare the function in `src/frontends/pytorch/src/op/op.hpp` (or in the new `.hpp` if complex):
```cpp
OutputVector translate_<new_op>(const NodeContext& context);
```

### 5b. 1:1 mapping (no custom translator needed)

When the PyTorch op maps directly to a single OV op with the same input/output arity and no attribute translation:

```cpp
// In op_table.cpp, no new .cpp file needed:
{"aten::abs", op::translate_1to1_match_1_inputs<opset1::Abs>},
{"aten.abs.default", op::translate_1to1_match_1_inputs<opset1::Abs>},
```

### 5c. Multi-input with type alignment

```cpp
OutputVector translate_add(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, false, false);
    return {context.mark_node(std::make_shared<v1::Add>(x, y))};
}
```

### 5d. Using shared helpers

```cpp
#include "utils.hpp"

OutputVector translate_layer_norm(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axes = context.const_input<std::vector<int64_t>>(1);

    // Utility: normalize axis values to positive range
    const auto rank = get_shape_rank(context, x);
    axes = normalize_axis(axes, rank);

    // Utility: make optional bias (returns Constant(0) if input is None)
    auto bias = make_optional_bias(context, x, 2);

    return { /* ... */ };
}
```

### 5e. Normalize-step transformations

For ops that need graph-level rewriting after translation:

```cpp
// In src/frontends/pytorch/src/transforms/<my_transform>.cpp:
class MyTransform : public ov::pass::MatcherPass {
public:
    MyTransform() {
        auto pattern = ov::pass::pattern::wrap_type<SequenceMark>();
        auto callback = [](ov::pass::pattern::Matcher& m) -> bool {
            // Pattern match + transform logic
            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MyTransform");
        register_matcher(m, callback);
    }
};

// Register in src/frontends/pytorch/src/frontend.cpp, in FrontEnd::normalize():
manager.register_pass<MyTransform>();
```

### 5f. Update `CMakeLists.txt`

In `src/frontends/pytorch/CMakeLists.txt`:
```cmake
set(SOURCES
    ...
    src/op/<new_op>.cpp
    ...
)
```

---

## 6. Adding Tests

### 6a. Create a Python layer test

In `tests/layer_tests/pytorch_tests/test_<op_name>.py`:

```python
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestMyOp(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, dim, keepdim):
        class MyModel(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, x):
                return torch.my_op(x, dim=self.dim, keepdim=self.keepdim)

        return MyModel(dim, keepdim), None, "my_op"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("keepdim", [True, False])
    def test_my_op(self, dim, keepdim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, keepdim), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dim", [0, 1])
    def test_my_op_torch_export(self, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, True), ie_device, precision, ir_version,
                   trace_model=True)
```

---

## 7. Build and Test

```bash
# Build the PyTorch frontend:
cmake --build build --target openvino_pytorch_frontend -j$(nproc)

# Run layer tests (TorchScript mode):
TEST_DEVICE=CPU TEST_PRECISION=FP32 python3 -m pytest tests/layer_tests/pytorch_tests/test_<op>.py -v -k "precommit"

# Run with torch.export mode:
TEST_DEVICE=CPU TEST_PRECISION=FP32 PYTORCH_TRACING_MODE=EXPORT python3 -m pytest tests/layer_tests/pytorch_tests/test_<op>.py -v -k "precommit_torch_export"

# Apply clang-format:
cmake --build build --target clang_format_fix_all -j$(nproc)
```

---

## 8. Validation Checklist

- [ ] **PyTorch schema consulted** — all inputs (including optional), constants, and attributes covered.
- [ ] **TorchScript mapping added** — registered in `get_supported_ops_ts()`.
- [ ] **FX/Export mapping added** — registered in `get_supported_ops_fx()` for all overload variants.
- [ ] **Inplace variant handled** — registered with `inplace_op<>` if applicable.
- [ ] **`mark_node()` called** — on every `std::make_shared<...>()` in the translator.
- [ ] **Optional inputs checked** — `input_is_none()` guard before each optional access.
- [ ] **`num_inputs_check()` called** — validates expected input count.
- [ ] **CMakeLists.txt updated** — new `.cpp` file added.
- [ ] **Layer test added** — `test_<op>.py` with `@pytest.mark.precommit` and `@pytest.mark.precommit_torch_export`.
- [ ] **Both tracing modes pass** — TorchScript and Export.
- [ ] **clang-format applied** — no formatting diffs.

---

## 9. Common Pitfalls

| Pitfall | Explanation |
|---|---|
| **Forgetting `mark_node()`** | Every `std::make_shared<...>()` must be wrapped with `context.mark_node()`. |
| **Not checking `input_is_none()`** | Optional inputs (e.g., bias, weight) may be None. Always guard before access. |
| **TorchScript vs FX name mismatch** | `aten::op_name` vs `aten.op_name.overload` — missing FX mapping causes silent `PtFrameworkNode` under `torch.export`. |
| **Missing inplace wrapper** | `aten::relu_` must use `inplace_op<translate_1to1_match_1_inputs<Relu>>`, not a separate translator. |
| **Wrong `const_input<T>()` type** | Using `const_input<int64_t>()` for a float attribute silently returns a wrong value. |
| **Not using `num_inputs_check()`** | Missing this guard leads to hard-to-debug out-of-bounds crashes when the model has fewer inputs. |
| **Not checking `common_translators`** | Check `src/frontends/common_translators/` before writing a new translator. |
| **Dynamic shape assumptions** | Translators must not assume static dimensions. Use `get_shape_rank()` and dynamic-safe OV ops. |
| **Missing pytest marks** | `@pytest.mark.precommit` and `@pytest.mark.precommit_torch_export` are required for CI gating. |

---

## 10. Reference Links

- [PyTorch Operator Documentation](https://pytorch.org/docs/stable/torch.html)
- [PyTorch ATen native_functions.yaml](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
- [torch.export Reference](https://pytorch.org/docs/stable/export.html)
- [OpenVINO Available Operations](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html)
- [OpenVINO PyTorch Frontend README](src/frontends/pytorch/README.md)

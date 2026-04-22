# Skill: Investigate and Fix PyTorch Model Conversion Issues in OpenVINO

---

## 1. Triage — Classify the Failure

Before diving into code, determine which category the issue falls into:

| Category | Symptoms |
|---|---|
| **Unsupported Op** | Error contains `"No translator found for"`, `PtFrameworkNode` remains in the converted graph, or a `torch.export` op is missing from `get_supported_ops_fx()`. |
| **Op Conversion Bug** | Model converts but inference accuracy is wrong (large numerical diff vs PyTorch), or `PYTORCH_OP_CONVERSION_CHECK` fires for a valid model. |
| **Shape/Type Mismatch** | Errors like `"Inconsistent element type"`, `"Shape mismatch"`, or partial-shape propagation failures after conversion. |
| **Tracing Mode Issue** | Model works with TorchScript but fails with `torch.export` (or vice versa). Op name mappings differ between `get_supported_ops_ts()` and `get_supported_ops_fx()`. |
| **Inplace Op** | Inplace variant (`aten::add_`, `aten.mul_.Tensor`) is missing or doesn't correctly wrap with `inplace_op<>`. |
| **Normalize-Step Failure** | Conversion succeeds but a post-conversion transformation in `FrontEnd::normalize()` crashes or produces wrong results. |
| **Complex / Sequence Type** | Errors related to `ComplexTypeMark` or `SequenceMark` — the Mark operation was not resolved during normalize. |

### Quick diagnostic commands

```bash
# Attempt conversion and print the error
python3 -c "
from openvino import convert_model
import torch

model = ...  # instantiate or load the model
example_input = torch.randn(1, 3, 224, 224)
try:
    ov_model = convert_model(model, example_input=example_input)
    print('Conversion OK, outputs:', [o.any_name for o in ov_model.outputs])
except Exception as e:
    print('ERROR:', e)
"

# Inspect FX graph ops (torch.export path):
python3 -c "
import torch
from torch.export import export
model = ...
em = export(model, (torch.randn(1, 3, 224, 224),))
for node in em.graph.nodes:
    if node.op == 'call_function':
        print(f'{node.target.__name__}')
" | sort | uniq -c | sort -rn
```

---

## 2. Key Source Locations

All paths are relative to the repository root.

### PyTorch Frontend Core

| Path | Description |
|---|---|
| `src/frontends/pytorch/src/op/` | **Per-op translators** — one `.cpp` file per PyTorch operation. This is where most fixes go. |
| `src/frontends/pytorch/src/op_table.cpp` | **Operator registry** — `get_supported_ops_ts()` (TorchScript) and `get_supported_ops_fx()` (torch.export/FX). |
| `src/frontends/pytorch/src/translate_session.cpp` | Main conversion loop. |
| `src/frontends/pytorch/src/frontend.cpp` | Entry point for `FrontEnd::convert()`, `convert_partially()`, and `normalize()`. |
| `src/frontends/pytorch/src/utils.hpp` | **Shared helpers** (`num_inputs_check`, `make_optional_bias`, `get_shape_rank`, `normalize_axis`, `numel`, `convert_dtype`, `concat_list_construct`). **Always check here first.** |
| `src/frontends/pytorch/include/openvino/frontend/pytorch/node_context.hpp` | `NodeContext` class — `get_input()`, `const_input<T>()`, `mark_node()`, `input_is_none()`. |
| `src/frontends/pytorch/src/pt_framework_node.hpp` | `PtFrameworkNode` — placeholder for unconverted ops. |
| `src/frontends/pytorch/src/transforms/` | **Normalize-step transformations** — graph passes that run after initial translation. |

### Cross-Frontend Shared Libraries

| Path | Description |
|---|---|
| `src/frontends/common_translators/include/common_translators.hpp` | **Reusable translators** shared across frontends. |
| `src/frontends/common/include/openvino/frontend/complex_type_mark.hpp` | `ComplexTypeMark` — Mark node for complex number type propagation. |
| `src/frontends/common/include/openvino/frontend/sequence_mark.hpp` | `SequenceMark` — Mark node for sequence/list type propagation. |

### Tests

| Path | Description |
|---|---|
| `tests/layer_tests/pytorch_tests/` | **Python layer tests** — one `test_<op>.py` file per operation. Primary test suite. |
| `tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py` | `PytorchLayerTest` base class. |
| `tests/model_hub_tests/pytorch/` | Model-level tests for HuggingFace Transformers, TorchVision, etc. |

### Build Targets

| Target | What it builds |
|---|---|
| `openvino_pytorch_frontend` | The PyTorch frontend shared library. |

---

## 3. Investigation Workflow

### Step 0: Reproduce the failure and identify the conversion path

| Path | How to identify | Op name format |
|---|---|---|
| **TorchScript** (default) | `torch.jit.script()` or `torch.jit.trace()` | `aten::add`, `aten::conv2d` |
| **torch.export** | `PYTORCH_TRACING_MODE=EXPORT` | `aten.add.Tensor`, `aten.conv2d.default` |

```bash
# Reproduce with TorchScript (default):
python3 -c "
from openvino import convert_model
import torch
model = ...
ov_model = convert_model(model, example_input=torch.randn(1, 3, 224, 224))
"

# Reproduce with torch.export:
PYTORCH_TRACING_MODE=EXPORT python3 -c "
from openvino import convert_model
import torch
model = ...
ov_model = convert_model(model, example_input=torch.randn(1, 3, 224, 224))
"
```

### Step 1: Identify the failing op

From the error message, extract the op name (e.g., `aten::grid_sampler` or `aten.grid_sampler_2d.default`).

```bash
# For TorchScript — list ops in the scripted graph:
python3 -c "
import torch
model = ...
scripted = torch.jit.script(model)
for node in scripted.inlined_graph.nodes():
    print(node.kind())
" | sort | uniq -c | sort -rn
```

### Step 2: Check if the op has a translator

```bash
# TorchScript ops — search in get_supported_ops_ts():
grep -n '"aten::<op_name>"' src/frontends/pytorch/src/op_table.cpp

# FX/Export ops — search in get_supported_ops_fx():
grep -n '"aten.<op_name>' src/frontends/pytorch/src/op_table.cpp

# Search for the translator implementation:
grep -rn 'translate_<op_name>' src/frontends/pytorch/src/op/
```

- No translator → see [add-fe-op/pytorch.md](../add-fe-op/pytorch.md)
- Found → go to Section 5 (Fixing an Existing Op)

### Step 3: Examine the translator code

```bash
cat src/frontends/pytorch/src/op/<op_name>.cpp
```

Cross-reference with [PyTorch docs](https://pytorch.org/docs/stable/) to check:
- Are all inputs handled (including optional ones via `input_is_none()`)?
- Is `num_inputs_check()` called to validate input count?
- Does the translator call `context.mark_node()` on every created node?

### Step 4: Check existing tests

```bash
ls tests/layer_tests/pytorch_tests/test_<op_name>.py
grep -rn '<op_name>' tests/layer_tests/pytorch_tests/ --include="*.py" | head -20
```

---

## 4. Adding a New Op Translator

For full implementation details, see [add-fe-op/pytorch.md](../add-fe-op/pytorch.md).

---

## 5. Fixing an Existing Op Translator

### Common fix patterns

#### A. Missing or wrong input handling

```cpp
// BEFORE (crashes if input is None):
auto bias = context.get_input(2);

// AFTER (check for None inputs):
if (!context.input_is_none(2)) {
    auto bias = context.get_input(2);
    result = context.mark_node(std::make_shared<v1::Add>(result, bias));
}
```

#### B. Missing FX/Export variant

1. Find the FX op name:
```bash
python3 -c "
import torch
from torch.export import export
em = export(model, (torch.randn(1, 3, 224, 224),))
for node in em.graph.nodes:
    if node.op == 'call_function':
        print(node.target.__name__)
"
```

2. Add the FX mapping in `get_supported_ops_fx()` in `op_table.cpp`:
```cpp
{"aten.op_name.default", op::translate_op_name},
```

#### C. Inplace op missing

```cpp
// In get_supported_ops_ts():
{"aten::relu_", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
// In get_supported_ops_fx():
{"aten.relu_.default", op::inplace_op<op::translate_1to1_match_1_inputs<opset10::Relu>>},
```

#### D. Shape/Type mismatch

- Use `align_eltwise_input_types` or `translate_1to1_match_2_inputs_align_types<>` for type alignment.
- Use `apply_dtype()` when the op needs a specific output dtype.

#### E. Numerical accuracy

```python
import torch, numpy as np
from openvino import convert_model, Core

model = ...
example = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pt_out = model(example)

ov_model = convert_model(model, example_input=example)
ov_out = Core().compile_model(ov_model, "CPU")(example.numpy())

ref = pt_out.detach().numpy()
abs_diff = np.abs(ref - list(ov_out.values())[0])
print(f"max_diff={abs_diff.max():.6e}, allclose={np.allclose(ref, list(ov_out.values())[0], atol=1e-5)}")
```

**Common accuracy root causes:**

| Cause | Fix |
|---|---|
| **Padding convention** | Check ceil_mode, auto-pad, floor/ceil rounding mode differences |
| **Epsilon placement** | Verify epsilon inside/outside sqrt matches PyTorch behavior |
| **Type promotion** | PyTorch may upcast to float64 implicitly; OV stays float32 |
| **Reduction axis** | Check 0-indexed vs negative indexing, keepdims defaults |

#### F. Normalize-step transformation issues

Transformations live in `src/frontends/pytorch/src/transforms/`. Temporarily disable passes to identify which one misbehaves.

---

## 6. Testing

```bash
# Run tests for a specific op:
TEST_DEVICE=CPU TEST_PRECISION=FP32 python3 -m pytest tests/layer_tests/pytorch_tests/test_<op_name>.py -v -k "precommit"

# Run with torch.export mode:
TEST_DEVICE=CPU TEST_PRECISION=FP32 PYTORCH_TRACING_MODE=EXPORT python3 -m pytest tests/layer_tests/pytorch_tests/test_<op_name>.py -v -k "precommit_torch_export"
```

---

## 7. Debugging Techniques

### Enable frontend verbose logging

```bash
export OV_LOG_LEVEL=DEBUG
python3 -c "from openvino import convert_model; import torch; convert_model(model, example_input=torch.randn(1,3,224,224))"
```

### Inspect the converted graph

```python
from openvino import convert_model, serialize
import torch

model = ...
ov_model = convert_model(model, example_input=torch.randn(1, 3, 224, 224))
for op in ov_model.get_ordered_ops():
    print(f"{op.get_type_name()} '{op.get_friendly_name()}'")
serialize(ov_model, "/tmp/model.xml", "/tmp/model.bin")
```

### Use partial conversion to find unconverted ops

```python
from openvino.frontend import FrontEndManager
import torch

fem = FrontEndManager()
fe = fem.load_by_framework("pytorch")
scripted = torch.jit.script(model)
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
decoder = TorchScriptPythonDecoder(scripted)
input_model = fe.load(decoder)
partial_model = fe.convert_partially(input_model)

for op in partial_model.get_ordered_ops():
    if "PtFrameworkNode" in op.get_type_name():
        print(f"UNCONVERTED: {op.get_friendly_name()}")
```

### GDB debugging

```bash
cmake -DCMAKE_BUILD_TYPE=Debug ...
cmake --build build --target openvino_pytorch_frontend -j$(nproc)
gdb --args python3 -m pytest tests/layer_tests/pytorch_tests/test_where.py -v -k "test_where[zeros-bool]" -s
# (gdb) break where.cpp:25
# (gdb) run
```

---

## 8. Checklist Before Submitting a Fix

- [ ] **Root cause identified** — the specific PyTorch op behavior that was violated or missing.
- [ ] **Minimal code change** — only translator, op_table, and test files are modified.
- [ ] **Python layer test added/updated** — in `tests/layer_tests/pytorch_tests/test_<op>.py`.
- [ ] **Both tracing modes covered** — TorchScript and FX/Export mappings present where applicable.
- [ ] **Inplace variant handled** — registered with `inplace_op<>`.
- [ ] **`mark_node()` called** — on every OpenVINO node created in the translator.
- [ ] **Build succeeds** — `cmake --build build --target openvino_pytorch_frontend` completes.
- [ ] **Layer tests pass** — both TorchScript and Export modes.
- [ ] **clang-format applied** — formatting changes committed.

---

## 9. Common Pitfalls

| Pitfall | Explanation |
|---|---|
| **Forgetting `mark_node()`** | Every `std::make_shared<...>()` in a translator must be wrapped in `context.mark_node()`. |
| **Not checking `input_is_none()`** | Optional inputs may be None. Always guard with `context.input_is_none(index)` first. |
| **TorchScript vs FX name mismatch** | TorchScript uses `aten::op_name`, FX uses `aten.op_name.overload`. Forgetting the FX mapping causes silent `PtFrameworkNode` under `torch.export`. |
| **Missing inplace wrapper** | Inplace ops need `op::inplace_op<base_translator>`. |
| **Wrong `const_input<T>()` type** | Produces silent wrong values. Check the PyTorch schema for the exact parameter type. |
| **Not using `num_inputs_check()`** | Skipping this leads to hard-to-debug out-of-bounds crashes. |
| **Not checking `common_translators`** | Check `src/frontends/common_translators/` before writing a new translator. |
| **Dynamic shape assumptions** | Translators must not assume static dimensions. |
| **Missing pytest marks** | New layer tests need `@pytest.mark.precommit` and `@pytest.mark.precommit_torch_export` for CI. |

---

## 10. Reference Links

- [PyTorch Operator Documentation](https://pytorch.org/docs/stable/torch.html)
- [torch.export Reference](https://pytorch.org/docs/stable/export.html)
- [OpenVINO Available Operations](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html)
- [OpenVINO PyTorch Frontend README](src/frontends/pytorch/README.md)

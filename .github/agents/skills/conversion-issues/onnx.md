# Skill: Investigate and Fix ONNX Model Conversion Issues in OpenVINO

---

## 1. Triage — Classify the Failure

Before diving into code, determine which category the issue falls into:

| Category | Symptoms |
|---|---|
| **Unsupported Op** | Error contains `"Not supported ONNX op"`, `ONNXFrameworkNode`, or `NotSupportedONNXNode`. The ONNX op has no translator registered. |
| **Op Conversion Bug** | Model converts but inference accuracy is wrong (large numerical diff), or an assertion like `CHECK_VALID_NODE` fires during conversion for a valid model. |
| **Shape/Type Mismatch** | Runtime errors such as `"Inconsistent element type"`, `"Shape mismatch"`, or partial-shape propagation failures. |
| **Opset Version Gap** | Error mentions an opset version, or the op was introduced/changed in a newer ONNX opset that OpenVINO hasn't implemented yet. |
| **External Data / Protobuf** | Errors about missing external weight files or protobuf parsing failures. |
| **Subgraph / Control Flow** | Errors inside `If`, `Loop`, `Scan` ops that carry sub-graphs. |

### Quick diagnostic commands

```bash
# Python: attempt conversion and print the error
python3 -c "
from openvino import Core
core = Core()
try:
    model = core.read_model('model.onnx')
    print('Conversion OK, outputs:', [o.any_name for o in model.outputs])
except Exception as e:
    print('ERROR:', e)
"

# Or using convert_model:
python3 -c "
from openvino import convert_model
model = convert_model('model.onnx')
print([o.any_name for o in model.outputs])
"

# Inspect ONNX model metadata:
python3 -c "
import onnx
m = onnx.load('model.onnx')
print('Opset imports:', [(o.domain or 'ai.onnx', o.version) for o in m.opset_import])
ops = set(n.op_type for n in m.graph.node)
print('Unique ops:', sorted(ops))
"
```

---

## 2. Key Source Locations

All paths are relative to the repository root.

### ONNX Frontend Core

| Path | Description |
|---|---|
| `src/frontends/onnx/frontend/src/op/` | **Per-op translators** — one `.cpp` file per ONNX operator. This is where most fixes go. |
| `src/frontends/onnx/frontend/src/op/com.microsoft/` | Translators for `com.microsoft` domain ops. |
| `src/frontends/onnx/frontend/src/op/org.openvinotoolkit/` | Translators for custom OpenVINO-specific ops. |
| `src/frontends/onnx/frontend/src/ops_bridge.cpp` | **Operator registry** — maps `(domain, op_name, opset_version)` → translator function. |
| `src/frontends/onnx/frontend/src/core/operator_set.hpp` | Defines `ONNX_OP` / `ONNX_OP_M` macros and `Operator` type alias. |
| `src/frontends/onnx/frontend/src/version_range.hpp` | `VersionRange`, `OPSET_RANGE`, `OPSET_SINCE`, `OPSET_IN` helpers. |
| `src/frontends/onnx/frontend/src/translate_session.cpp` | Main conversion loop. |
| `src/frontends/onnx/frontend/src/frontend.cpp` | Entry point for `FrontEnd::convert()`, `load()`, `convert_partially()`. |
| `src/frontends/onnx/frontend/src/exceptions.hpp` | `CHECK_VALID_NODE` macro and `OnnxNodeValidationFailure` exception. |
| `src/frontends/onnx/frontend/src/core/node.hpp` | `ov::frontend::onnx::Node` — context object each translator receives. |
| `src/frontends/onnx/frontend/src/utils/common.hpp` | Shared helpers (`is_input_valid`, `handle_opset6_binary_op`). **Always check here first.** |

### Cross-Frontend Shared Libraries

| Path | Description |
|---|---|
| `src/frontends/common_translators/include/common_translators.hpp` | **Reusable translators** shared across ONNX/PyTorch/TF frontends. |
| `src/frontends/common_translators/src/op/` | Implementation of common translators. |
| `src/frontends/common/include/openvino/frontend/complex_type_mark.hpp` | `ComplexTypeMark` — Mark node for complex number type propagation. |
| `src/frontends/common/include/openvino/frontend/sequence_mark.hpp` | `SequenceMark` — Mark node for sequence/list type propagation. |
| `src/frontends/common_translators/include/sequence_concat_replacer.hpp` | Normalize-step transformation example. |

### ONNX Frontend Tests

| Path | Description |
|---|---|
| `src/frontends/onnx/tests/onnx_import.in.cpp` | Main C++ import/inference tests (templated per backend). |
| `src/frontends/onnx/tests/onnx_import_com_microsoft.in.cpp` | Tests for `com.microsoft` domain. |
| `src/frontends/onnx/tests/onnx_import_controlflow.in.cpp` | Tests for ops with sub-graphs (`If`, `Loop`, `Scan`). |
| `src/frontends/onnx/tests/onnx_import_convpool.in.cpp` | Convolution and pooling op tests. |
| `src/frontends/onnx/tests/models/` | `.prototxt` test model definitions. |

### Build Targets

| Target | What it builds |
|---|---|
| `openvino_onnx_frontend` | The ONNX frontend shared library. |
| `ov_onnx_frontend_tests` | C++ unit/import tests. |

---

## 3. Investigation Workflow

### Step 0: Check if the model works with ONNX Runtime (CPU provider)

```python
import numpy as np
import onnxruntime as ort
import onnx

model = onnx.load("model.onnx")
opset_versions = [(o.domain or "ai.onnx", o.version) for o in model.opset_import]
print(f"Opset imports: {opset_versions}")

try:
    sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
    inputs = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
    result = sess.run(None, inputs)
    print(f"ORT inference OK, {len(result)} outputs")
except Exception as e:
    print(f"ORT inference FAILED: {e}")
```

**Decision based on ORT result:**

| ORT Result | Model Age | Action |
|---|---|---|
| **ORT fails** | Old model (opset ≤ 9) | Ask the user to consider not supporting this model. |
| **ORT fails** | Newer model (opset ≥ 13) | The model may be corrupted. Ask user to verify with `onnx.checker.check_model()`. |
| **ORT succeeds** | Any | Valid model — the issue is in OpenVINO's ONNX frontend. |

### Step 1: Reproduce the failure

```bash
python3 -c "
from openvino import Core
import numpy as np
core = Core()
model = core.read_model('model.onnx')
compiled = core.compile_model(model, 'CPU')
result = compiled(np.zeros((1, 3, 224, 224), dtype=np.float32))
print('Output shapes:', {k: v.shape for k, v in result.items()})
"
```

### Step 2: Identify the failing op

```bash
python3 -c "
import onnx
m = onnx.load('model.onnx')
for n in m.graph.node:
    print(f'{n.op_type} (domain={n.domain or \"ai.onnx\"})')
" | sort | uniq -c | sort -rn
```

If the error names the op, skip to Step 3. Otherwise use partial conversion:

```bash
python3 -c "
from openvino.frontend import FrontEndManager
from openvino import serialize
fem = FrontEndManager()
fe = fem.load_by_framework('onnx')
model = fe.load('model.onnx')
ov_model = fe.convert_partially(model)
serialize(ov_model, '/tmp/partial.xml')
"
grep -i 'framework\|not.supported' /tmp/partial.xml
```

### Step 3: Check if the op has a translator

```bash
# Example: looking for "Resize"
grep -rn 'ONNX_OP.*"Resize"' src/frontends/onnx/frontend/src/op/
```

- No result → **unsupported op** → see [add-fe-op/onnx.md](../add-fe-op/onnx.md)
- Found → **existing translator** → go to Section 5 (Fixing an Existing Op)

Check the registered opset range:

```bash
grep -n 'ONNX_OP' src/frontends/onnx/frontend/src/op/resize.cpp
```

### Step 4: Examine the translator code

```bash
cat src/frontends/onnx/frontend/src/op/<op_name>.cpp
```

Cross-reference with the [ONNX operator spec](https://onnx.ai/onnx/operators/) to check:
- Are all attributes handled with correct defaults per the ONNX spec?
- Are optional inputs guarded with `common::is_input_valid(node, index)`?
- Are edge cases covered (empty tensors, scalar inputs, negative indices)?
- Does the opset version introduce new attributes or behavior changes?

### Step 5: Check existing tests

```bash
grep -rn '<op_name>' src/frontends/onnx/tests/onnx_import*.in.cpp
grep -rn '<op_name>' src/frontends/onnx/tests/models/*.prototxt | head -20
```

---

## 4. Adding a New Op Translator

For full implementation details, see [add-fe-op/onnx.md](../add-fe-op/onnx.md).

---

## 5. Fixing an Existing Op Translator

### Common fix patterns

#### A. Missing attribute or default value

```cpp
// BEFORE (crashes if attribute missing):
const auto axis = node.get_attribute_value<int64_t>("axis");

// AFTER (provide default per ONNX spec):
const auto axis = node.get_attribute_value<int64_t>("axis", 0);
```

#### B. Unhandled optional input

```cpp
#include "utils/common.hpp"

const auto inputs = node.get_ov_inputs();
if (common::is_input_valid(node, 2)) {
    const auto& bias = inputs[2];
    result = std::make_shared<v1::Add>(result, bias);
}
```

#### C. Opset version upgrade (new attributes/behavior)

Add a new opset namespace and register it:

```cpp
namespace opset_18 {
ov::OutputVector op_name(const ov::frontend::onnx::Node& node) {
    // Handle new opset-18 attributes/behavior
}
ONNX_OP("OpName", OPSET_SINCE(18), ai_onnx::opset_18::op_name);
}  // namespace opset_18
```

Update the old registration's upper bound:

```cpp
// Change OPSET_SINCE(11) to OPSET_RANGE(11, 17):
ONNX_OP("OpName", OPSET_RANGE(11, 17), ai_onnx::opset_11::op_name);
```

#### D. Shape/Type mismatch

Ensure the translator handles dynamic dimensions, different element types, broadcasting rules per ONNX spec, and scalar vs 1-D tensor differences.

#### E. Numerical accuracy

If inference works but results are wrong:
1. Isolate the failing op using `onnx.utils.extract_model()`.
2. Compare per-op outputs between ONNX Runtime and OpenVINO.
3. Check if OV op semantics differ from ONNX (padding conventions, rounding modes, epsilon placement).

#### F. Multi-op patterns — normalize-step transformations

When a fix requires matching and replacing a pattern of multiple ops:
1. Write an `ov::pass::MatcherPass` transformation.
2. Register it in `FrontEnd::normalize()` (in `frontend.cpp`) via the `pass::Manager`.
3. For cross-frontend transformations, place it in `src/frontends/common_translators/`.

---

## 6. Testing the Fix

```bash
# Build and run all ONNX frontend tests
cmake --build build --target ov_onnx_frontend_tests -j$(nproc)
cd build && ctest -R "ov_onnx_frontend_tests" --output-on-failure -j$(nproc)

# Run a specific test by name pattern
ctest -R "onnx_model_<test_name>" -V
```

### Validate with Python comparison

```python
import numpy as np
import onnxruntime as ort
from openvino import Core

sess = ort.InferenceSession("model.onnx")
ort_inputs = {"input": np.random.randn(1, 3, 224, 224).astype(np.float32)}
ort_out = sess.run(None, ort_inputs)

core = Core()
model = core.read_model("model.onnx")
compiled = core.compile_model(model, "CPU")
ov_out = compiled(ort_inputs)

for i, (ref, ov_val) in enumerate(zip(ort_out, ov_out.values())):
    max_diff = np.abs(ref - ov_val).max()
    print(f"Output {i}: max_diff={max_diff:.6e}, allclose={np.allclose(ref, ov_val, atol=1e-5, rtol=1e-5)}")
```

---

## 7. Debugging Techniques

### Enable ONNX frontend verbose logging

```bash
export OV_LOG_LEVEL=DEBUG
python3 -c "from openvino import Core; Core().read_model('model.onnx')"
```

### Inspect the model graph post-conversion

```python
from openvino import Core, serialize

core = Core()
model = core.read_model("model.onnx")
for op in model.get_ordered_ops():
    print(f"{op.get_type_name()} '{op.get_friendly_name()}' "
          f"inputs={[str(i.get_partial_shape()) for i in op.inputs()]} "
          f"outputs={[str(o.get_partial_shape()) for o in op.outputs()]}")
serialize(model, "/tmp/model.xml", "/tmp/model.bin")
```

### Use partial conversion to find unconverted ops

```python
from openvino.frontend import FrontEndManager

fem = FrontEndManager()
fe = fem.load_by_framework("onnx")
input_model = fe.load("model.onnx")
partial_model = fe.convert_partially(input_model)

for op in partial_model.get_ordered_ops():
    if "FrameworkNode" in op.get_type_name():
        attrs = op.get_attrs()
        print(f"UNCONVERTED: {attrs.get('ONNX_META_type', '?')} domain={attrs.get('ONNX_META_domain', '')}")
```

### Extract a minimal failing subgraph

```python
import onnx

model = onnx.load("model.onnx")
model = onnx.shape_inference.infer_shapes(model)
onnx.utils.extract_model("model.onnx", "submodel.onnx",
                          input_names=["input_tensor_name"],
                          output_names=["problematic_op_output_name"])
```

### GDB debugging for C++ translator issues

```bash
cmake --preset debug
cmake --build build --target ov_onnx_frontend_tests -j$(nproc)
gdb --args ./build/bin/ov_onnx_frontend_tests --gtest_filter="*onnx_model_test_name*"
# (gdb) break resize.cpp:65
# (gdb) run
```

---

## 8. Checklist Before Submitting a Fix

- [ ] **Root cause identified** — the specific ONNX spec requirement that was violated or missing.
- [ ] **Minimal code change** — only translator file(s) and test(s) are modified.
- [ ] **Test model added** — `.prototxt` in `src/frontends/onnx/tests/models/` exercising the fix.
- [ ] **C++ test added** — in the appropriate `onnx_import*.in.cpp` file.
- [ ] **Build succeeds** — `cmake --build build --target openvino_onnx_frontend ov_onnx_frontend_tests` completes.
- [ ] **All ONNX tests pass** — `ctest -R ov_onnx_frontend_tests` shows no regressions.
- [ ] **clang-format applied** — `cmake --build build --target clang_format_fix_all` run.
- [ ] **Supported ops doc** — verify `src/frontends/onnx/docs/supported_ops.md` reflects new op.
- [ ] **Opset range correct** — `ONNX_OP` covers the right version range per ONNX changelog.
- [ ] **Edge cases** — dynamic shapes, optional inputs, unusual dtypes, scalars handled.

---

## 9. Common Pitfalls

| Pitfall | Explanation |
|---|---|
| **Forgetting optional inputs** | Use `common::is_input_valid(node, index)` for all optional input checks. |
| **Wrong opset version range** | Use `OPSET_RANGE(1, 10)` + `OPSET_SINCE(11)` if spec changed behavior in opset 11. |
| **Attribute type mismatch** | Using the wrong `get_attribute_value<T>()` silently returns a default. |
| **Broadcasting assumptions** | OV's Numpy-style broadcasting may differ from ONNX in edge cases. |
| **Not handling empty tensors** | Some ops may receive 0-element tensors. Ensure no division by size. |
| **Modifying the wrong opset function** | When fixing for opset 18, create a new function — don't modify the opset 11 one. |
| **Missing `#include`** | Each OpenVINO op needs its own include (e.g., `#include "openvino/op/add.hpp"`). |
| **Not checking common_translators** | Check `src/frontends/common_translators/` before writing a new translator. |
| **Not using Mark operations** | For complex, sequence, or quantized types, use `ComplexTypeMark`/`SequenceMark`. |

---

## 10. Reference Links

- [ONNX Operator Specifications](https://onnx.ai/onnx/operators/)
- [ONNX Operator Changelog](https://github.com/onnx/onnx/blob/main/docs/Changelog.md)
- [OpenVINO Available Operations](https://docs.openvino.ai/latest/openvino_docs_ops_opset.html)
- [OpenVINO ONNX Frontend README](src/frontends/onnx/README.md)
- [OpenVINO ONNX Supported Ops](src/frontends/onnx/docs/supported_ops.md)

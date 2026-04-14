# Skill: FE Op Analysis

> Source: `skills/add-fe-op/SKILL.md` (Step 1)
> Agent: `fe_agent`

## When to Use

- Model conversion fails with a missing translator or unsupported op error from a FE.
- An operation is absent from the FE op mapping table.
- Partial support exists (translator file present but registration missing, or vice versa).

## Procedure

### 1. Identify the operation and source framework

From the error context / conversion log, extract:

- Exact op name:
  - PyTorch: `aten::op_name` (TorchScript) or `aten.op_name.default` (FX)
  - TensorFlow: `tf.nn.op_name` or raw op string (e.g. `GatherNd`)
  - ONNX: op string (e.g. `ScatterND`, versioned by opset)
- Source framework: `pytorch`, `tensorflow`, or `onnx`
- Version/variant where applicable (ONNX opset, TorchScript vs FX namespace)

### 2. Check current support state

Browse or clone `https://github.com/openvinotoolkit/openvino` and run:

**PyTorch:**
```bash
# Translator file
ls src/frontends/pytorch/src/op/<op_name>.cpp

# TorchScript key
grep -n 'aten::<op_name>' src/frontends/pytorch/src/op_table.cpp

# FX key
grep -n 'aten\.<op_name>' src/frontends/pytorch/src/op_table.cpp
```

**TensorFlow:**
```bash
ls src/frontends/tensorflow/src/op/<op_name>.cpp
grep -n '<OpName>' src/frontends/tensorflow/src/op_table.cpp
```

**ONNX:**
```bash
ls src/frontends/onnx/src/ops/<op_name>.cpp
grep -rn 'ONNX_OP.*"<OpName>"' src/frontends/onnx/src/ops/
```

Support states:

| State | Meaning |
|-------|---------|
| `full` | Translator file + registration both present → skip scaffolding |
| `partial` | One piece missing → repair only that part |
| `missing` | Nothing exists → full scaffold needed |

### 3. Research the op semantics

Consult official documentation for:

- Math formula / semantics
- Inputs, outputs, attributes
- Data types and broadcasting rules
- Edge cases (empty tensors, negative indices, etc.)

| Framework | Reference |
|-----------|-----------|
| PyTorch | https://pytorch.org/docs/stable/generated/torch.<op>.html |
| TensorFlow | https://www.tensorflow.org/api_docs/python/tf/ |
| ONNX | https://onnx.ai/onnx/operators/ |

### 4. Check for equivalent OV core op

Search existing OV ops:

```bash
grep -r '<OpName>\|<semantic_keyword>' \
  openvino/src/core/include/openvino/op/ \
  openvino/src/core/include/openvino/opsets/
```

Decision table:

| Situation | Action |
|-----------|--------|
| 1:1 match with existing OV op | Real translation — proceed |
| Composition of existing OV ops produces correct semantics | Real translation — proceed |
| No match; decomposing is too slow or inaccurate | **Escalate to Core Agent** |
| Operation is clearly decomposable via graph transform | Decomposable — flag for Transformation Agent |

### 5. Check cross-frontend reuse

If the same op exists in another frontend, reuse its logic rather than duplicating:

```bash
grep -rn '<op_name>' src/frontends/ --include="*.cpp"
```

If a working implementation found in another FE, extract shared translation
logic into a common utility header instead of copy-pasting.

## Output

Return a structured analysis block:

```
op_name:           <name>
source_framework:  <pytorch|tensorflow|onnx>
support_state:     <full|partial|missing>
translator_file:   <path or "missing">
registration:
  torchscript_key: <yes|no>   # PyTorch only
  fx_key:          <yes|no>   # PyTorch only
  onnx_op_macro:   <yes|no>   # ONNX only
  tf_table_entry:  <yes|no>   # TF only
math_formula:      <brief description>
inputs:            [<name: type>, ...]
outputs:           [<name: type>, ...]
attributes:        [<name: type = default>, ...]
ov_equivalent:     <op name(s) or "none">
cross_fe_reuse:    <yes|no> — <source path if yes>
action:            <translate|repair|escalate_to_core>
reason:            <why this action was chosen>
```

### Action routing

- `action=translate` or `action=repair` → proceed to **fe_op_translation** skill.
- `action=escalate_to_core` → fill the full escalation payload (see `fe_agent.md`)
  and stop. Do not proceed to Translation skill.

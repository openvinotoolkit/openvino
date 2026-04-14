# Skill: FE Op Testing

> Source: `skills/add-fe-op/SKILL.md` (Steps 4, 5, 6)
> Agent: `fe_agent`

## Prerequisites

- Completed **fe_op_registration** — translator file and registration entries are in place.

---

## Test Files to Create

### PyTorch layer test

File: `tests/layer_tests/pytorch_tests/test_<op_name>.py`

```python
import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestOpName(PytorchLayerTest):
    def _prepare_input(self, shape):
        return (np.random.randn(*shape).astype(np.float32),)

    def create_model(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.<op_name>(x)

        return Model(), None, ["aten::<op_name>"]

    @pytest.mark.parametrize("shape", [[2, 4], [1, 3, 5]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_<op_name>(self, shape, ie_device, precision, ir_version):
        self._test(self.create_model(), self._prepare_input(shape), ie_device, precision, ir_version)


class TestOpNameFX(PytorchLayerTest):
    """Same test through the FX / torch.export path."""

    def _prepare_input(self):
        return (np.random.randn(2, 4, 4).astype(np.float32),)

    def create_model(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.<op_name>(x)

        return Model(), None, ["aten.<op_name>.default"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_<op_name>_fx(self, ie_device, precision, ir_version):
        self._test(self.create_model(), self._prepare_input(), ie_device, precision, ir_version)
```

### TensorFlow layer test

File: `tests/layer_tests/tensorflow_tests/test_tf_<op_name>.py`

Follow naming and structure conventions from existing TF layer tests in the same
directory. Use `TensorFlowLayerTest` base class.

### ONNX frontend smoke test

File: `src/frontends/onnx/tests/test_<op_name>.cpp` (not applicable to PyTorch)

Use the existing test harness (`onnx_test_utils.hpp`) for ONNX.

---

## Conversion Validation

### When OV is installed and FE changes are included

```python
import openvino as ov
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.<op_name>(x)

model = Model()
dummy = torch.randn(2, 4, 4)
ov_model = ov.convert_model(model, example_input=dummy)
compiled = ov.compile_model(ov_model, "CPU")
result = compiled({"x": dummy.numpy()})
print(f"Conversion OK — output shape: {result[0].shape}")
```

Run this script. If it produces OV graph nodes (not framework fallback nodes),
report `validation=pass`.

### When OV is not installed with FE changes

> **Do NOT build OpenVINO.** Compilation takes too long on GHA nodes.

Mark validation as `blocked`:

```
validation: blocked — installed OV package does not include the patched FE.
                      Conversion check cannot run. Patch is provided for manual review.
```

**Critical:** Never report `success` when conversion was tested against a
package that does not contain the FE patch.

---

## Git Patch Generation

After translator, registration, and test files are written, generate a
`git format-patch` from the openvino working tree.

```bash
# Stage all FE changes
git add src/frontends/<frontend>/src/op/<op_name>.cpp
git add src/frontends/<frontend>/src/op_table.cpp
git add src/frontends/<frontend>/CMakeLists.txt
git add tests/layer_tests/<frontend>_tests/test_<op_name>.py

# Commit locally (do NOT push to openvino)
git commit -m "feat(fe/<frontend>): add translator for <op_name>"

# Export patch
git format-patch HEAD~1 --stdout > ../patches/fe_<op_name>_<frontend>.patch

# Verify it applies cleanly to a fresh checkout
git apply --check ../patches/fe_<op_name>_<frontend>.patch
echo "Patch check exit code: $?"
```

Place final patch in: `patches/fe_<op_name>_<frontend>.patch`

---

## Post Patch as GitHub Issue Comment

Use the existing script:

```bash
python scripts/post_issue_comment.py \
  --issue "$TICKET_NUMBER" \
  --title "FE patch: <op_name> (<frontend>)" \
  --body "$(cat patches/fe_<op_name>_<frontend>.patch)"
```

Include in the comment body:

- Patch header (files changed, lines added/removed)
- Validation status: `pass | fail | blocked`
- If `blocked`: explicit reason (e.g. installed OV does not include FE patch)
- If `partial` (fallback stub used): clear note that real OV op is still needed

---

## Reporting States

| State | Meaning |
|-------|---------|
| `success` | Real translation in place, conversion produces OV graph nodes, tests written |
| `partial` | Fallback stub in place — model does not crash, but no real OV graph yet |
| `blocked` | Validation could not run (installed OV does not contain FE patch) |
| `failed` | Translator or test produced errors |

---

## Output

- Test files created and ready for commit.
- Git patch file: `patches/fe_<op_name>_<frontend>.patch`
- Patch posted to the tracking GitHub issue as a comment.
- Validation status reported back to FE Agent.

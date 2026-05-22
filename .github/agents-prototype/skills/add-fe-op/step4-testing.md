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

After writing translator and registration, run the end-to-end conversion check using the
**`verify-conversion`** skill:

> Read and follow: `skills/verify-conversion/SKILL.md`

The skill auto-detects the conversion path (optimum-intel / ovc / convert_model) and
runs a numerical sanity check. Report the outcome (`validation=pass|fail|blocked`) back
to the FE Agent before generating the patch.

---

## Git Patch Generation

After translator, registration, and test files are written, generate a patch file.
Run:

```
python .github/scripts/meat/generate_patch.py --component fe_<frontend> --op <op_name>
```

Place the resulting `.patch` file in `agent-results/frontend/patches/`.

---

## Save Patch and Report

Write the patch path and validation status to the FE Agent output file:

```
python .github/scripts/meat/record_result.py --agent frontend --op <op_name> --patch <patch_path> --validation <pass|fail|blocked>
```

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
- Patch available in `agent-results/pytorch-fe/patches/`.
- Validation status reported back to FE Agent.

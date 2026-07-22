---
name: python-bootstrap
description: Install Python dependencies before running verification or test steps. Choose the correct path depending on whether the agent builds OpenVINO from source or not.
---

# Skill: Python Package Bootstrap

Install packages **only when they are actually needed** for a step. Do not
pre-install everything upfront.

---

## Path A — No source build (core-opspec, transformation, frontend, npu)

Use this path when the agent **does not** compile OpenVINO from source.
The `openvino` release wheel provides the runtime needed for tests and
conversion checks.

```bash
pip install openvino optimum-intel torch \
    --extra-index-url https://download.pytorch.org/whl/cpu
```

For frontend agents that also need ONNX or TensorFlow:
```bash
pip install onnx onnxruntime
pip install tensorflow  # only if testing TF frontend
```

---

## Path B — Source build (cpu, gpu)

Use this path when the agent **builds OpenVINO from source**.

> **Do NOT `pip install openvino`** when a source build is present in the
> same environment. Installing the release wheel alongside a dev build creates
> two `openvino` packages that shadow each other, producing confusing import
> errors and incorrect test results.

Install only the packages that are **not** produced by the OV build:

```bash
pip install torch \
    --extra-index-url https://download.pytorch.org/whl/cpu
pip install optimum-intel pytest
# Do NOT add 'openvino' here — use the build output instead
```

After building OpenVINO, add the build's Python bindings to the path:

```bash
# Point Python at the locally built openvino package
export PYTHONPATH="<build_dir>/src/bindings/python:$PYTHONPATH"
# or install the wheel produced by the build
pip install <build_dir>/wheels/openvino-*.whl
```

---

## General rules

- Install packages **on demand** per step, not all at once.
- If a package install fails, note it in `agent-results/<agent>/session.md`
  and continue — do not abort the entire pipeline over a missing optional dep.
- Never install from untrusted or unofficial indexes beyond the PyTorch
  CPU wheel server listed above.

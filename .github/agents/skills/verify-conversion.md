---
name: verify-conversion
description: E2E gate — verifies that applied patches produce a working, numerically sane end-to-end inference through the OpenVINO plugin. Handles HuggingFace/optimum-intel, native OV conversion (ovc/convert_model), and ONNX. Used by orchestrators as the mandatory gate before any PR is published.
---

# Skill: Verify Conversion (E2E Gate)

> **This skill is a hard gate before PR publication.**
> A PR must **not** be opened until both steps below pass:
> 1. The model converts without error.
> 2. A real end-to-end inference through the OV plugin layer produces numerically
>    sane output (no NaN/Inf, non-empty, correct shape).
>
> Conversion success alone is not sufficient — plugin-level issues (wrong kernel
> output, silent data corruption, incorrect type/shape inference at runtime) are
> only caught by an actual inference run.

This is **not** a full strategy matrix run — use `try-conversion` for that.

Goal: one conversion, one inference run, numerical sanity check, structured result.

---

## Step 1 — Determine conversion path

Read `agent-results/pipeline_state.json` (or context file) to identify:

| Signal | Conversion path |
|---|---|
| `model_id` starts with `org/name` (HuggingFace ID) AND `optimum_supported=true` | **Path A** — `optimum-cli` |
| `model_id` is a local path with `.onnx` file | **Path B** — OV native (`ovc` / `convert_model`) |
| `model_id` is a local PyTorch model or `.pt` / `.pth` file | **Path B** — OV native (`convert_model`) |
| `model_id` is a local TF SavedModel or `.pb` file | **Path B** — OV native (`ovc`) |
| Unclear | Try Path A first, fall back to Path B on failure |

---

## Path A — HuggingFace model via optimum-intel

### Auto-detect task

```python
from transformers import AutoConfig

PIPELINE_TAG_MAP = {
    "text-generation": "text-generation-with-past",
    "text2text-generation": "text2text-generation-with-past",
    "image-text-to-text": "image-text-to-text",
    "text-classification": "text-classification",
    "token-classification": "token-classification",
    "question-answering": "question-answering",
    "feature-extraction": "feature-extraction",
    "fill-mask": "fill-mask",
    "text-to-image": "text-to-image",
    "image-to-text": "image-to-text",
    "automatic-speech-recognition": "automatic-speech-recognition",
    "audio-classification": "audio-classification",
}
try:
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    pipeline_tag = getattr(cfg, "pipeline_tag", None)
    model_type = getattr(cfg, "model_type", "")
    if pipeline_tag:
        task = PIPELINE_TAG_MAP.get(pipeline_tag, pipeline_tag)
    elif model_type in ("t5", "mt5", "bart", "mbart"):
        task = "text2text-generation-with-past"
    else:
        task = "text-generation-with-past"
except Exception:
    task = "text-generation-with-past"
print(f"[verify] Resolved task: {task}")
```

### Export

```bash
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task "$TASK" \
  --weight-format fp16 \
  ov_verify_check/
```

If export fails with a timeout or OOM, retry with `--weight-format int4`.

### Quick inference check

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("ov_verify_check/", trust_remote_code=True)
model = OVModelForCausalLM.from_pretrained("ov_verify_check/", trust_remote_code=True)
inputs = tok("Hello", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=5)
print("[verify] Inference OK:", tok.decode(out[0]))
```

For non-causal models (classification, ASR, etc.), adapt the class and inputs accordingly.

---

## Path B — Native OV conversion (`ovc` / `convert_model`)

### ONNX model

```python
import openvino as ov
import numpy as np
core = ov.Core()
model = core.read_model("path/to/model.onnx")
compiled = core.compile_model(model, "CPU")
# Run one inference pass
infer = compiled.create_infer_request()
for inp in compiled.inputs:
    shape = [d if d > 0 else 1 for d in inp.partial_shape.get_min_shape()]
    infer.set_tensor(inp, ov.Tensor(np.zeros(shape, dtype=inp.element_type.to_dtype())))
infer.infer()
print("[verify] ONNX compile + inference OK")
```

### PyTorch model

```python
import torch, openvino as ov

# Load your torch model
# torch_model = ...

example_input = torch.zeros(1, 3, 224, 224)   # adjust shape
ov_model = ov.convert_model(torch_model, example_input=example_input)
compiled = ov.Core().compile_model(ov_model, "CPU")
result = list(compiled({0: example_input.numpy()}).values())[0]
print("[verify] PyTorch convert + inference OK, output shape:", result.shape)
```

### TF / generic via ovc

```bash
ovc path/to/saved_model --output_model ov_verify_check/model.xml
python3 -c "
import openvino as ov, numpy as np
core = ov.Core()
m = core.read_model('ov_verify_check/model.xml')
cmp = core.compile_model(m, 'CPU')
infer = cmp.create_infer_request()
for inp in cmp.inputs:
    shape = [d if d > 0 else 1 for d in inp.partial_shape.get_min_shape()]
    infer.set_tensor(inp, ov.Tensor(np.zeros(shape, dtype=inp.element_type.to_dtype())))
infer.infer()
print('[verify] ovc + compile + inference OK')
"
```

---

## Step 3 — E2E Numerical Sanity Check

After inference completes, validate output quality through the plugin layer:

```python
import numpy as np

def check_output_sanity(outputs: dict, label: str) -> tuple[bool, str]:
    """Returns (passed, reason). Checks all output tensors."""
    for name, arr in outputs.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            return False, f"{label}: output '{name}' is empty (size=0)"
        if np.isnan(arr).any():
            return False, f"{label}: output '{name}' contains NaN"
        if np.isinf(arr).any():
            return False, f"{label}: output '{name}' contains Inf"
    return True, "OK"

# For HF/optimum path — check that generated tokens are non-empty
def check_lm_output(out_ids, tokenizer, label: str) -> tuple[bool, str]:
    if out_ids is None or out_ids.shape[-1] == 0:
        return False, f"{label}: generated token sequence is empty"
    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    if not decoded.strip():
        return False, f"{label}: decoded output is blank"
    return True, f"generated: '{decoded[:80]}'"
```

Apply the appropriate check based on conversion path:
- **Path A (HF/optimum)**: use `check_lm_output` on the generated token ids
- **Path B (native OV)**: use `check_output_sanity` on the infer request output tensors

If the sanity check fails, set `e2e_passed = false` with the reason — do **not**
silently swallow the failure.

---

## Result reporting

Write the outcome to `agent-results/<agent>/verify_result.json`:

```json
{
  "verify_passed": true,
  "e2e_passed": true,
  "conversion_path": "optimum-cli | ovc | convert_model",
  "task": "<task or null>",
  "e2e_detail": "<short description of what was run and what output was checked>",
  "error": null
}
```

`verify_passed` is `true` only when **both** conversion and E2E inference pass.

On failure, set `"verify_passed": false`, `"e2e_passed": false` (if inference
failed), and populate `"error"` with the specific failure reason.

Do **not** abort the pipeline — the orchestrator decides whether to retry or
escalate.

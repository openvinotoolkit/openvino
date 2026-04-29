---
name: verify-conversion
description: Quick post-fix sanity check — verify a model converts and produces output with OpenVINO. Handles HuggingFace/optimum-intel, native OV conversion (ovc/convert_model), and ONNX. Used by orchestrators after applying patches.
---

# Skill: Verify Conversion

Quick sanity check to confirm that applied patches produce a working conversion.
This is **not** a full strategy matrix run — use `try-conversion` for that.

Goal: one conversion attempt, one inference pass, pass/fail result.

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
core = ov.Core()
model = core.read_model("path/to/model.onnx")
compiled = core.compile_model(model, "CPU")
print("[verify] ONNX read + compile OK")
```

### PyTorch model

```python
import torch, openvino as ov

# Load your torch model
# torch_model = ...

example_input = torch.zeros(1, 3, 224, 224)   # adjust shape
ov_model = ov.convert_model(torch_model, example_input=example_input)
compiled = ov.Core().compile_model(ov_model, "CPU")
result = compiled(example_input.numpy())
print("[verify] PyTorch convert + inference OK")
```

### TF / generic via ovc

```bash
ovc path/to/saved_model --output_model ov_verify_check/model.xml
python3 -c "
import openvino as ov
m = ov.Core().read_model('ov_verify_check/model.xml')
ov.Core().compile_model(m, 'CPU')
print('[verify] ovc + compile OK')
"
```

---

## Result reporting

Write the outcome to `agent-results/<agent>/verify_result.json`:

```json
{
  "verify_passed": true,
  "conversion_path": "optimum-cli | ovc | convert_model",
  "task": "<task or null>",
  "error": null
}
```

On failure, set `"verify_passed": false` and populate `"error"` with the last
traceback line. Do **not** abort the pipeline — the orchestrator decides whether
to retry or escalate.

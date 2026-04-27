---
name: classify-failure
description: Classify each failed conversion attempt into the 9-class error taxonomy and extract routing signals for the orchestrator. Maps tracebacks to root causes, identifies specific components, and produces a machine-readable signal block.
---

# Skill: Classify Failure

**Trigger:** Called after `try-conversion` when no strategy succeeded, or after
a successful conversion that fails inference. Reads `conversion_attempts.json`
and `model_profile.json`.

---

## Error Taxonomy

Map every failure traceback to exactly one class:

| Class | Key signatures | First specialist |
|-------|---------------|-----------------|
| `optimum_unsupported_arch` | `KeyError` from `TasksManager`, `model_type` not registered | Optimum-Intel |
| `optimum_export_bug` | Exception inside `optimum/exporters/`, wrongly-shaped dummy inputs, `TypeError` in model config | Optimum-Intel |
| `missing_model_dependency` | `ModuleNotFoundError`, `ImportError: requires package`, `pip install` hint in error | Optimum-Intel |
| `missing_conversion_rule` | `NotImplemented` in PyTorch FE, `aten::*` op not covered, `ConversionError: No rule` | OV Orchestrator (FE Agent) |
| `frontend_error` | Exception inside `openvino/frontend/`, segfault during tracing, IR parse error | OV Orchestrator (FE Agent) |
| `ir_validation_error` | Shape inference failure, `ngraph::validate_and_infer_types`, invalid IR structure | OV Orchestrator |
| `inference_runtime_error` | Exception from OV runtime, plugin error, `ov::Exception` during infer | OV Orchestrator |
| `genai_unsupported` | `ValueError` from `openvino_genai`, missing chat template, pipeline construction error | GenAI |
| `tokenizer_error` | Exception from `openvino_tokenizers`, tokenizer conversion failure, SentencePiece error | Tokenizers |
| `unknown_arch_transformers_too_old` | `model_type` absent from installed transformers but visible on HF Hub | Optimum-Intel (transformers upgrade first) |
| `unknown` | None of the above match | Optimum-Intel (fallback) |

---

## Step 1 — Load Artifacts

```python
import json, re

with open("conversion_attempts.json") as f:
    attempts = json.load(f)

with open("model_profile.json") as f:
    profile = json.load(f)

# Collect all error text from failed attempts
all_errors = []
for a in attempts:
    if not a["success"]:
        combined = (a.get("stderr", "") + "\n" + a.get("stdout", "")).strip()
        all_errors.append({"id": a["id"], "text": combined})
```

---

## Step 2 — Pattern Matching

```python
def classify_error(text: str) -> str:
    t = text.lower()

    # Order matters — most specific first
    if re.search(r"modulenotfounderror|importerror.*requires.*package|no module named", t):
        return "missing_model_dependency"

    if re.search(r"keyerror.*taskmanager|model_type.*not.*support|no configuration class", t):
        # Distinguish: is model_type missing from transformers or from optimum?
        if re.search(r"no.*class.*for.*model_type|transformers.*doesn.t.*know", t):
            return "unknown_arch_transformers_too_old"
        return "optimum_unsupported_arch"

    if re.search(r"optimum[/\\]exporters|dummy_inputs|onnx_config|ModelPatcher|"
                 r"from_pretrained.*export|openvino_config", t):
        return "optimum_export_bug"

    if re.search(r"notimplemented.*aten::|no rule.*op|conversion.*failed.*aten|"
                 r"pytorch_frontend|pt_frontend|torchscript.*error", t):
        return "missing_conversion_rule"

    if re.search(r"openvino[/\\]frontend|ir.*parse|frontend.*error|"
                 r"segmentation fault|core dumped", t):
        return "frontend_error"

    if re.search(r"validate_and_infer_types|ngraph.*shape|ir.*invalid|"
                 r"shape inference failed|opset.*mismatch", t):
        return "ir_validation_error"

    if re.search(r"ov::exception|infer.*request|plugin.*error|"
                 r"openvino.*runtime.*error|inference engine", t):
        return "inference_runtime_error"

    if re.search(r"openvino_genai|chat.*template.*missing|pipeline.*construct", t):
        return "genai_unsupported"

    if re.search(r"openvino_tokenizers|sentencepiece|tokenizer.*convert|"
                 r"tiktoken.*error|fast.*tokenizer", t):
        return "tokenizer_error"

    return "unknown"


# Classify each failed attempt
for err in all_errors:
    err["error_class"] = classify_error(err["text"])
    # Extract first meaningful traceback line
    lines = err["text"].splitlines()
    tb_lines = [l for l in lines if
                "Error" in l or "Exception" in l or "raise " in l]
    err["key_line"] = tb_lines[0].strip() if tb_lines else lines[-1].strip() if lines else ""

# Dominant class = class of the last (most sophisticated) attempt
dominant_class = all_errors[-1]["error_class"] if all_errors else "unknown"
print(f"Dominant error class: {dominant_class}")
```

---

## Step 3 — Extract Routing Signals

```python
def extract_signals(attempts, profile, dominant_class):
    signals = {
        "error_class": dominant_class,
        "requires_optimum_new_arch": False,
        "requires_transformers_upgrade": False,
        "transformers_override": "",
        "requires_tokenizer_check": False,
        "trust_remote_code_required": profile["trust_remote_code_required"],
        "is_vlm": profile["is_vlm"],
        "custom_ops_suspected": False,
        "oom_suspected": False,
        "target_agent": "",  # filled below
    }

    all_text = " ".join(
        a.get("stderr", "") + a.get("stdout", "") for a in attempts
    ).lower()

    # Optimum arch signals
    if dominant_class in ("optimum_unsupported_arch", "unknown_arch_transformers_too_old"):
        signals["requires_optimum_new_arch"] = True

    if dominant_class == "unknown_arch_transformers_too_old":
        signals["requires_transformers_upgrade"] = True
        signals["transformers_override"] = (
            "git+https://github.com/huggingface/transformers.git"
        )

    # Custom ops / recurrent patterns in config
    ssm_keys = [k for k in profile.get("special_config_keys", [])
                if any(w in k.lower() for w in
                       ["ssm", "mamba", "rwkv", "recurrent", "conv", "delta"])]
    if ssm_keys:
        signals["custom_ops_suspected"] = True

    # OOM signals
    if re.search(r"out of memory|oom|killed|memory error|cannot allocate", all_text):
        signals["oom_suspected"] = True

    # VLM implies tokenizer check after export
    if profile["is_vlm"]:
        signals["requires_tokenizer_check"] = True

    # Determine target agent
    routing = {
        "optimum_unsupported_arch": "optimum-intel",
        "optimum_export_bug": "optimum-intel",
        "missing_model_dependency": "optimum-intel",
        "unknown_arch_transformers_too_old": "optimum-intel",
        "unknown": "optimum-intel",
        "missing_conversion_rule": "enable-operator",
        "frontend_error": "enable-operator",
        "ir_validation_error": "enable-operator",
        "inference_runtime_error": "enable-operator",
        "genai_unsupported": "openvino-genai",
        "tokenizer_error": "openvino-tokenizers",
    }
    signals["target_agent"] = routing.get(dominant_class, "optimum-intel")

    return signals


signals = extract_signals(attempts, profile, dominant_class)
print("Routing signals:", json.dumps(signals, indent=2))

with open("routing_signals.json", "w") as f:
    json.dump(signals, f, indent=2)
```

---

## Step 4 — Extract the Key Error Excerpt

For each failed attempt, extract the most diagnostic ~20 lines to include in
the report — the traceback tail plus context lines before the final exception.

```python
def extract_excerpt(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    # Find last traceback block
    tb_start = max(
        (i for i, l in enumerate(lines) if "Traceback (most recent call last)" in l),
        default=0
    )
    excerpt = lines[tb_start:]
    return "\n".join(excerpt[-max_lines:])


excerpts = {}
for err in all_errors:
    excerpts[err["id"]] = extract_excerpt(err["text"])

with open("error_excerpts.json", "w") as f:
    json.dump(excerpts, f, indent=2)

print(f"Excerpts saved for {len(excerpts)} failed attempt(s).")
```

---

## Output

| File | Contents |
|------|----------|
| `routing_signals.json` | Machine-readable signals for orchestrator routing |
| `error_excerpts.json` | Key traceback excerpts per attempt |

Key signal fields consumed by orchestrator:

| Signal | Meaning |
|--------|---------|
| `error_class` | 11-value taxonomy class |
| `target_agent` | Recommended first specialist |
| `requires_optimum_new_arch` | Optimum-Intel must add full model support |
| `requires_transformers_upgrade` | Transformers version is the root cause |
| `transformers_override` | Pip install string for git-HEAD transformers |
| `custom_ops_suspected` | SSM/recurrent ops detected — likely needs `_ov_ops.py` |
| `is_vlm` | Vision-language model — extra routing needed |
| `oom_suspected` | OOM hit — suggest int4 or sharded export |

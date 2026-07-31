# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Classify-failure step for the analyze-and-convert agent.

Reads conversion_attempts.json and model_profile.json, classifies each
failure into the 11-class error taxonomy, extracts routing signals, and
writes error excerpts for use by build_report.py.

Usage:
    python .github/scripts/meat/classify_failure.py

Input files (current working directory):
    conversion_attempts.json
    model_profile.json

Output files:
    routing_signals.json
    error_excerpts.json
"""

import json
import pathlib
import re
import sys

# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------

def load_json(path: str, default):
    p = pathlib.Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[classify_failure] WARNING: could not parse {path}: {exc}", file=sys.stderr)
    else:
        print(f"[classify_failure] WARNING: {path} not found — using default", file=sys.stderr)
    return default


attempts = load_json("conversion_attempts.json", [])
profile  = load_json("model_profile.json", {})

# Collect error text from failed attempts
all_errors = []
for a in attempts:
    if not a.get("success"):
        combined = (a.get("stderr", "") + "\n" + a.get("stdout", "")).strip()
        all_errors.append({"id": a["id"], "text": combined})


# ---------------------------------------------------------------------------
# Step 2 — Pattern matching → 11-class taxonomy
# ---------------------------------------------------------------------------

def classify_error(text: str) -> str:
    t = text.lower()

    # Order matters — most specific first
    if re.search(r"modulenotfounderror|importerror.*requires.*package|no module named", t):
        return "missing_model_dependency"

    if re.search(r"keyerror.*taskmanager|model_type.*not.*support|no configuration class", t):
        if re.search(r"no.*class.*for.*model_type|transformers.*doesn.t.*know", t):
            return "unknown_arch_transformers_too_old"
        return "optimum_unsupported_arch"

    if re.search(
        r"optimum[/\\]exporters|dummy_inputs|onnx_config|ModelPatcher|"
        r"from_pretrained.*export|openvino_config",
        t,
    ):
        return "optimum_export_bug"

    if re.search(
        r"notimplemented.*aten::|no rule.*op|conversion.*failed.*aten|"
        r"pytorch_frontend|pt_frontend|torchscript.*error",
        t,
    ):
        return "missing_conversion_rule"

    if re.search(
        r"openvino[/\\]frontend|ir.*parse|frontend.*error|"
        r"segmentation fault|core dumped",
        t,
    ):
        return "frontend_error"

    if re.search(
        r"validate_and_infer_types|ngraph.*shape|ir.*invalid|"
        r"shape inference failed|opset.*mismatch",
        t,
    ):
        return "ir_validation_error"

    if re.search(
        r"ov::exception|infer.*request|plugin.*error|"
        r"openvino.*runtime.*error|inference engine",
        t,
    ):
        return "inference_runtime_error"

    if re.search(r"openvino_genai|chat.*template.*missing|pipeline.*construct", t):
        return "genai_unsupported"

    if re.search(
        r"openvino_tokenizers|sentencepiece|tokenizer.*convert|"
        r"tiktoken.*error|fast.*tokenizer",
        t,
    ):
        return "tokenizer_error"

    return "unknown"


for err in all_errors:
    err["error_class"] = classify_error(err["text"])
    lines = err["text"].splitlines()
    tb_lines = [l for l in lines if "Error" in l or "Exception" in l or "raise " in l]
    err["key_line"] = tb_lines[0].strip() if tb_lines else (lines[-1].strip() if lines else "")

dominant_class = all_errors[-1]["error_class"] if all_errors else "unknown"
print(f"[classify_failure] Dominant error class: {dominant_class}")


# ---------------------------------------------------------------------------
# Step 3 — Extract routing signals
# ---------------------------------------------------------------------------

def extract_signals(attempts, profile, dominant_class):
    signals = {
        "error_class": dominant_class,
        "requires_optimum_new_arch": False,
        "requires_transformers_upgrade": False,
        "transformers_override": "",
        "requires_tokenizer_check": False,
        "trust_remote_code_required": profile.get("trust_remote_code_required", False),
        "is_vlm": profile.get("is_vlm", False),
        "custom_ops_suspected": False,
        "oom_suspected": False,
        "target_agent": "",
    }

    all_text = " ".join(
        a.get("stderr", "") + a.get("stdout", "") for a in attempts
    ).lower()

    if dominant_class in ("optimum_unsupported_arch", "unknown_arch_transformers_too_old"):
        signals["requires_optimum_new_arch"] = True

    if dominant_class == "unknown_arch_transformers_too_old":
        signals["requires_transformers_upgrade"] = True
        signals["transformers_override"] = (
            "git+https://github.com/huggingface/transformers.git"
        )

    ssm_keys = [
        k for k in profile.get("special_config_keys", [])
        if any(w in k.lower() for w in
               ["ssm", "mamba", "rwkv", "recurrent", "conv", "delta"])
    ]
    if ssm_keys:
        signals["custom_ops_suspected"] = True

    if re.search(r"out of memory|oom|killed|memory error|cannot allocate", all_text):
        signals["oom_suspected"] = True

    if profile.get("is_vlm"):
        signals["requires_tokenizer_check"] = True

    routing = {
        "optimum_unsupported_arch":        "optimum-intel",
        "optimum_export_bug":              "optimum-intel",
        "missing_model_dependency":        "optimum-intel",
        "unknown_arch_transformers_too_old": "optimum-intel",
        "unknown":                         "optimum-intel",
        "missing_conversion_rule":         "enable-operator",
        "frontend_error":                  "enable-operator",
        "ir_validation_error":             "enable-operator",
        "inference_runtime_error":         "enable-operator",
        "genai_unsupported":               "openvino-genai",
        "tokenizer_error":                 "openvino-tokenizers",
    }
    signals["target_agent"] = routing.get(dominant_class, "optimum-intel")

    return signals


signals = extract_signals(attempts, profile, dominant_class)
print("[classify_failure] Routing signals:", json.dumps(signals, indent=2))

pathlib.Path("routing_signals.json").write_text(
    json.dumps(signals, indent=2), encoding="utf-8"
)
print("[classify_failure] Written: routing_signals.json")


# ---------------------------------------------------------------------------
# Step 4 — Extract key error excerpts
# ---------------------------------------------------------------------------

def extract_excerpt(text: str, max_lines: int = 20) -> str:
    lines = text.splitlines()
    tb_start = max(
        (i for i, l in enumerate(lines) if "Traceback (most recent call last)" in l),
        default=0,
    )
    excerpt = lines[tb_start:]
    return "\n".join(excerpt[-max_lines:])


excerpts = {err["id"]: extract_excerpt(err["text"]) for err in all_errors}

pathlib.Path("error_excerpts.json").write_text(
    json.dumps(excerpts, indent=2), encoding="utf-8"
)
print(f"[classify_failure] Written: error_excerpts.json ({len(excerpts)} attempt(s))")

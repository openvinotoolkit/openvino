# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Try-conversion step for the analyze-and-convert agent.

Attempts model export to OpenVINO IR using a systematic strategy matrix.
Tries multiple weight formats, optimum-intel versions, and flag combinations.
Records every attempt with full output in conversion_attempts.json.

Usage:
    python .github/scripts/meat/try_conversion.py

Input files (current working directory):
    model_profile.json

Output files:
    conversion_attempts.json
    ov_model_<strategy_id>/   — IR files from first successful strategy
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Load profile
# ---------------------------------------------------------------------------

profile_path = pathlib.Path("model_profile.json")
if not profile_path.exists():
    print("[try_conversion] ERROR: model_profile.json not found. Run probe_model.py first.",
          file=sys.stderr)
    sys.exit(1)

profile = json.loads(profile_path.read_text(encoding="utf-8"))

MODEL_ID          = profile["model_id"]
TRUST_RC          = profile.get("trust_remote_code_required", False)
IS_VLM            = profile.get("is_vlm", False)
OPTIMUM_SUPPORTED = profile.get("optimum_supported", True)

print(f"[try_conversion] model_id        : {MODEL_ID}")
print(f"[try_conversion] trust_rc        : {TRUST_RC}")
print(f"[try_conversion] is_vlm          : {IS_VLM}")
print(f"[try_conversion] optimum_support : {OPTIMUM_SUPPORTED}")

# ---------------------------------------------------------------------------
# Step 1 — Determine task
# ---------------------------------------------------------------------------

PIPELINE_TAG_MAP = {
    "text-generation":                  "text-generation-with-past",
    "text2text-generation":             "text2text-generation-with-past",
    "image-text-to-text":               "image-text-to-text",
    "text-classification":              "text-classification",
    "token-classification":             "token-classification",
    "question-answering":               "question-answering",
    "feature-extraction":               "feature-extraction",
    "fill-mask":                        "fill-mask",
    "text-to-image":                    "text-to-image",
    "image-to-text":                    "image-to-text",
    "automatic-speech-recognition":     "automatic-speech-recognition",
    "audio-classification":             "audio-classification",
    "zero-shot-image-classification":   "zero-shot-image-classification",
}

task = "text-generation-with-past"  # safe default
try:
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    pipeline_tag = getattr(cfg, "pipeline_tag", None) or profile.get("pipeline_tag")
    if pipeline_tag is None:
        model_type = getattr(cfg, "model_type", "")
        task = (
            "text2text-generation-with-past"
            if model_type in ("t5", "mt5", "bart", "mbart")
            else "text-generation-with-past"
        )
    else:
        task = PIPELINE_TAG_MAP.get(pipeline_tag, pipeline_tag)
except Exception as exc:  # noqa: BLE001
    print(f"[try_conversion] WARNING: could not resolve task from config: {exc}",
          file=sys.stderr)

print(f"[try_conversion] resolved task: {task}")

# ---------------------------------------------------------------------------
# Step 2 — Build strategy matrix
# ---------------------------------------------------------------------------

strategies = []

_GIT_OPTIMUM = "git+https://github.com/huggingface/optimum-intel.git"
_GIT_TRANSFORMERS = "git+https://github.com/huggingface/transformers.git"

# Strategy A: fp16, stable optimum-intel
strategies.append({
    "id":               "A-fp16-stable",
    "weight_format":    "fp16",
    "optimum_version":  "stable",
    "extra_flags":      [],
    "description":      "fp16, stable optimum-intel",
})

# Strategy B: int8 symmetric, stable (lighter on memory)
strategies.append({
    "id":               "B-int8-stable",
    "weight_format":    "int8",
    "optimum_version":  "stable",
    "extra_flags":      [],
    "description":      "int8, stable optimum-intel",
})

# Strategy C: fp16, optimum-intel git-HEAD
strategies.append({
    "id":               "C-fp16-gitHEAD",
    "weight_format":    "fp16",
    "optimum_version":  _GIT_OPTIMUM,
    "extra_flags":      [],
    "description":      "fp16, optimum-intel git-HEAD",
})

# Strategy D: int4 AWQ — only for large models (>7B)
if profile.get("estimated_params_b", 0) > 7:
    strategies.append({
        "id":               "D-int4-awq-gitHEAD",
        "weight_format":    "int4",
        "optimum_version":  _GIT_OPTIMUM,
        "extra_flags":      ["--ratio", "1.0", "--group-size", "128"],
        "description":      "int4 AWQ, optimum-intel git-HEAD (large model)",
    })

# Strategy E: fp16 + git-HEAD optimum + transformers pre-release
strategies.append({
    "id":                    "E-fp16-transformers-pre",
    "weight_format":         "fp16",
    "optimum_version":       _GIT_OPTIMUM,
    "extra_flags":           [],
    "description":           "fp16, git-HEAD + transformers pre-release",
    "transformers_override": _GIT_TRANSFORMERS,
})

# ---------------------------------------------------------------------------
# Step 3 — Run each strategy
# ---------------------------------------------------------------------------

attempts = []

for s in strategies:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Strategy {s['id']}: {s['description']}")
    print(sep)

    out_dir = f"ov_model_{s['id']}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Install required optimum-intel version when non-stable.
    # Set _TRY_CONVERSION_SKIP_PIP=1 to suppress installs (e.g. in tests).
    _skip_pip = os.environ.get("_TRY_CONVERSION_SKIP_PIP", "")
    if s["optimum_version"] != "stable" and not _skip_pip:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", s["optimum_version"]],
            check=False,
        )

    if s.get("transformers_override") and not _skip_pip:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", s["transformers_override"]],
            check=False,
        )

    cmd = [
        "optimum-cli", "export", "openvino",
        "--model", MODEL_ID,
        "--task", task,
        "--weight-format", s["weight_format"],
        "--output", out_dir,
    ]
    if TRUST_RC:
        cmd.append("--trust-remote-code")
    cmd.extend(s.get("extra_flags", []))

    t0 = time.time()
    # shell=True is required on Windows so that cmd.exe resolves the tool via
    # PATHEXT (real optimum-cli installs as .exe; test stubs may be .bat/.cmd).
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        shell=(sys.platform == "win32"),
    )
    elapsed = round(time.time() - t0, 1)

    attempt = {
        "id":           s["id"],
        "description":  s["description"],
        "weight_format": s["weight_format"],
        "optimum_version": s["optimum_version"],
        "command":      " ".join(cmd),
        "returncode":   proc.returncode,
        "elapsed_s":    elapsed,
        "stdout":       proc.stdout[-8000:],
        "stderr":       proc.stderr[-8000:],
        "success":      False,
        "ir_files":     [],
    }

    if proc.returncode == 0:
        ir_files = [
            f for f in os.listdir(out_dir)
            if f.endswith(".xml") or f.endswith(".bin")
        ] if os.path.isdir(out_dir) else []

        if ir_files:
            attempt["success"]  = True
            attempt["ir_files"] = ir_files
            attempt["ir_dir"]   = out_dir
            print(f"  SUCCESS in {elapsed}s — IR files: {ir_files}")
        else:
            attempt["stderr"] += "\n[No IR files found in output dir despite exit 0]"
            print(f"  EXIT 0 but no IR files produced in {out_dir}")
    else:
        print(f"  FAILED (rc={proc.returncode}) in {elapsed}s")
        last_err = (proc.stderr or proc.stdout)[-500:]
        print(f"  Last error: {last_err}")

    attempts.append(attempt)

    if attempt["success"]:
        print(f"\n[try_conversion] First successful strategy: {s['id']}")
        break

pathlib.Path("conversion_attempts.json").write_text(
    json.dumps(attempts, indent=2), encoding="utf-8"
)
print(f"\n[try_conversion] Attempts recorded : {len(attempts)}")
print(f"[try_conversion] Successful        : {sum(1 for a in attempts if a['success'])}")

# ---------------------------------------------------------------------------
# Step 4 — Quick inference sanity check (on success only)
# ---------------------------------------------------------------------------

winning = next((a for a in attempts if a.get("success")), None)

if winning is None:
    print("[try_conversion] No successful conversion — skipping inference check.")
else:
    ir_dir = winning["ir_dir"]
    print(f"[try_conversion] Running inference check on: {ir_dir}")
    try:
        from optimum.intel import OVModelForCausalLM  # type: ignore
        from transformers import AutoTokenizer  # type: ignore

        tok   = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=TRUST_RC)
        model = OVModelForCausalLM.from_pretrained(ir_dir, trust_remote_code=TRUST_RC)

        inputs    = tok("Hello, world!", return_tensors="pt")
        out       = model.generate(**inputs, max_new_tokens=10)
        generated = tok.decode(out[0], skip_special_tokens=True)

        print(f"[try_conversion] Inference OK: '{generated}'")
        winning["inference_ok"]     = True
        winning["inference_sample"] = generated
    except Exception as exc:  # noqa: BLE001
        print(f"[try_conversion] Inference check failed: {exc}", file=sys.stderr)
        winning["inference_ok"]    = False
        winning["inference_error"] = str(exc)

    # Persist updated inference results
    pathlib.Path("conversion_attempts.json").write_text(
        json.dumps(attempts, indent=2), encoding="utf-8"
    )

print("[try_conversion] Written: conversion_attempts.json")

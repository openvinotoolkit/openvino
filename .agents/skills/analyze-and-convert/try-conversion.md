---
name: try-conversion
description: Attempt model export to OpenVINO IR using a systematic strategy matrix. Tries multiple weight formats, optimum-intel versions, and flag combinations. Records every attempt with full output.
---

# Skill: Try Conversion

**Trigger:** Called after `probe-model` produces `model_profile.json`.
Uses the profile to select and execute the right strategy matrix.

---

## Prerequisites

```bash
# Isolated venv per conversion session
python -m venv venv-convert
source venv-convert/bin/activate   # Linux/macOS
# or: venv-convert\Scripts\activate  # Windows

pip install -q openvino optimum[openvino] huggingface_hub
```

Load profile:
```python
import json
with open("model_profile.json") as f:
    profile = json.load(f)

MODEL_ID = profile["model_id"]
TRUST_RC = profile["trust_remote_code_required"]
IS_VLM   = profile["is_vlm"]
OPTIMUM_SUPPORTED = profile["optimum_supported"]
```

---

## Step 1 — Determine Task

```python
# Resolve pipeline_tag → optimum-cli export task
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
    "zero-shot-image-classification": "zero-shot-image-classification",
}
try:
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    pipeline_tag = getattr(cfg, "pipeline_tag", None)
    if pipeline_tag is None:
        model_type = getattr(cfg, "model_type", "")
        task = "text2text-generation-with-past" if model_type in ("t5", "mt5", "bart", "mbart") else "text-generation-with-past"
    else:
        task = PIPELINE_TAG_MAP.get(pipeline_tag, pipeline_tag)
except Exception:
    task = "text-generation-with-past"
print(f"Resolved task: {task}")
```

---

## Step 2 — Build Strategy Matrix

Each strategy is a dict of `optimum-cli` flags to try.
Ordered from fastest/cheapest to most permissive.

```python
strategies = []

# Base flags shared by all strategies
base = {
    "--model": MODEL_ID,
    "--task": task,
    "--output": "ov_model",
}
if TRUST_RC:
    base["--trust-remote-code"] = ""

# Strategy A: fp16, stable optimum-intel
strategies.append({
    "id": "A-fp16-stable",
    "weight_format": "fp16",
    "optimum_version": "stable",   # current installed
    "extra_flags": [],
    "description": "fp16, stable optimum-intel",
})

# Strategy B: int8 symmetric, stable (lighter on memory)
strategies.append({
    "id": "B-int8-stable",
    "weight_format": "int8",
    "optimum_version": "stable",
    "extra_flags": [],
    "description": "int8, stable optimum-intel",
})

# Strategy C: fp16, optimum-intel git-HEAD
strategies.append({
    "id": "C-fp16-gitHEAD",
    "weight_format": "fp16",
    "optimum_version": "git+https://github.com/huggingface/optimum-intel.git",
    "extra_flags": [],
    "description": "fp16, optimum-intel git-HEAD",
})

# Strategy D: int4 AWQ (if model is large — > 7B)
if profile.get("estimated_params_b", 0) > 7:
    strategies.append({
        "id": "D-int4-awq-gitHEAD",
        "weight_format": "int4",
        "optimum_version": "git+https://github.com/huggingface/optimum-intel.git",
        "extra_flags": ["--ratio", "1.0", "--group-size", "128"],
        "description": "int4 AWQ, optimum-intel git-HEAD (large model)",
    })

# Strategy E: transformers override (if git-HEAD also fails, try pinning to latest pre-release)
strategies.append({
    "id": "E-fp16-transformers-pre",
    "weight_format": "fp16",
    "optimum_version": "git+https://github.com/huggingface/optimum-intel.git",
    "extra_flags": [],
    "description": "fp16, git-HEAD + transformers pre-release",
    "transformers_override": "git+https://github.com/huggingface/transformers.git",
})
```

---

## Step 3 — Run Each Strategy

```python
import os, subprocess, sys, time, shutil, json

attempts = []

for s in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy {s['id']}: {s['description']}")
    print(f"{'='*60}")

    # Fresh output dir per attempt
    out_dir = f"ov_model_{s['id']}"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Install required optimum-intel version if different
    if s["optimum_version"] != "stable":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", s["optimum_version"]],
            check=False
        )

    if s.get("transformers_override"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", s["transformers_override"]],
            check=False
        )

    # Build command
    cmd = ["optimum-cli", "export", "openvino",
           "--model", MODEL_ID,
           "--task", task,
           "--weight-format", s["weight_format"],
           "--output", out_dir]

    if TRUST_RC:
        cmd.append("--trust-remote-code")

    cmd.extend(s.get("extra_flags", []))

    # Run with full output capture
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - t0, 1)

    attempt = {
        "id": s["id"],
        "description": s["description"],
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "stdout": proc.stdout[-8000:],   # cap to last 8 KB
        "stderr": proc.stderr[-8000:],
        "success": False,
        "ir_files": [],
    }

    if proc.returncode == 0:
        # Verify IR files were actually produced
        ir_files = [f for f in os.listdir(out_dir)
                    if f.endswith(".xml") or f.endswith(".bin")]
        if ir_files:
            attempt["success"] = True
            attempt["ir_files"] = ir_files
            attempt["ir_dir"] = out_dir
            print(f"  SUCCESS in {elapsed}s — IR files: {ir_files}")
        else:
            attempt["stderr"] += "\n[No IR files found in output dir despite exit 0]"
            print(f"  EXIT 0 but no IR files produced.")
    else:
        print(f"  FAILED (rc={proc.returncode}) in {elapsed}s")
        print(f"  Last error: {(proc.stderr or proc.stdout)[-500:]}")

    attempts.append(attempt)

    # Stop on first success
    if attempt["success"]:
        print(f"\nFirst successful strategy: {s['id']}")
        break

# Persist results
with open("conversion_attempts.json", "w") as f:
    json.dump(attempts, f, indent=2)

print(f"\nAttempts recorded: {len(attempts)}")
print(f"Successful: {sum(1 for a in attempts if a['success'])}")
```

---

## Step 4 — Quick Inference Sanity Check (on success)

Only runs when at least one strategy succeeded.

```python
import json

with open("conversion_attempts.json") as f:
    attempts = json.load(f)

winning = next((a for a in attempts if a["success"]), None)
if not winning:
    print("No successful conversion — skipping inference check.")
else:
    ir_dir = winning["ir_dir"]
    print(f"Running inference check on: {ir_dir}")
    try:
        from optimum.intel import OVModelForCausalLM
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = OVModelForCausalLM.from_pretrained(ir_dir, trust_remote_code=True)

        inputs = tok("Hello, world!", return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=10)
        generated = tok.decode(out[0], skip_special_tokens=True)
        print(f"Inference OK: '{generated}'")
        winning["inference_ok"] = True
        winning["inference_sample"] = generated
    except Exception as e:
        print(f"Inference check failed: {e}")
        winning["inference_ok"] = False
        winning["inference_error"] = str(e)

    with open("conversion_attempts.json", "w") as f:
        json.dump(attempts, f, indent=2)
```

---

## Output

| File | Contents |
|------|----------|
| `conversion_attempts.json` | All strategy attempts with commands, full stdout/stderr, success flag, IR file list |
| `ov_model_<strategy_id>/` | IR files from the first successful strategy (if any) |

If **no strategy succeeded**, `conversion_attempts.json` contains the complete
failure evidence for `classify-failure` to analyse.

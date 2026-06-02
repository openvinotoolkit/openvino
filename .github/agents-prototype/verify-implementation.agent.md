---
name: Verify Implementation Agent
description: Build and test verification gate. Called by Enable Operator Agent after all coding agents complete. Builds changed OpenVINO components, runs their unit tests, and performs a quick inference sanity check if an IR model is available. Returns pass/fail with details before the E2E conversion gate.
model: claude-sonnet-4.6
---
# Verify Implementation Agent

## Role

Post-coding verification gate. Ensures that every change produced by the coding
agents (FE, Core OpSpec, Transformation, CPU, GPU) actually compiles, passes unit
tests, and does not crash at inference time.

This agent does **not** add new code. It only builds, runs existing tests, and reports.

## Output

Write all results to `agent-results/verify-implementation/`.

## Called by

- **Enable Operator Agent** (Phase 5 — after all parallel agents and FE Final Pass complete,
  before the E2E Verification Gate)

---

## Environment

| Item | Notes |
|---|---|
| **OpenVINO repository** | Current working directory — run from the `openvinotoolkit/openvino` repository root |
| **Build directory** | Use the existing `build/` directory — do **not** reconfigure CMake |

---

## Execution Model

### Step 1: Identify Changed Components

Read all sub-agent result files to determine which build targets to check:

```python
import json, os, glob

result_files = glob.glob("agent-results/*/.*_result.json") + \
               glob.glob("agent-results/**/*_result.json", recursive=True)

changed_components = set()
patch_files = []

for path in result_files:
    try:
        with open(path) as f:
            data = json.load(f)
        patches = data.get("patch_paths", []) or ([data["patch_path"]] if data.get("patch_path") else [])
        patch_files.extend(patches)
    except (json.JSONDecodeError, KeyError, FileNotFoundError):
        continue

# Map patch file paths to CMake build targets
target_map = {
    "src/frontends/pytorch":        "openvino_pytorch_frontend",
    "src/frontends/onnx":           "openvino_onnx_frontend",
    "src/frontends/tensorflow":     "openvino_tensorflow_frontend",
    "src/common/transformations":   "ov_transformations",
    "src/core":                     "openvino",
    "src/plugins/intel_cpu":        "openvino_intel_cpu_plugin",
    "src/plugins/intel_gpu":        "openvino_intel_gpu_plugin",
}

test_target_map = {
    "src/frontends/pytorch":      ("ov_pytorch_frontend_tests",    "ov_pytorch_frontend_tests"),
    "src/frontends/onnx":         ("ov_onnx_frontend_tests",       "ov_onnx_frontend_tests"),
    "src/frontends/tensorflow":   ("ov_tensorflow_frontend_tests", "ov_tensorflow_frontend_tests"),
    "src/common/transformations": ("ov_transformations_tests",     "ov_transformations_tests"),
    "src/core":                   ("ov_core_unit_tests",           "ov_core_unit_tests"),
    "src/plugins/intel_cpu":      ("ov_cpu_func_tests",            "ov_cpu_func_tests"),
}

for patch in patch_files:
    for prefix, target in target_map.items():
        if prefix in str(patch):
            changed_components.add(prefix)

print("Changed components:", sorted(changed_components))
```

If no patch files are found, default to checking all components that have
`agent-results/<component>/*_result.json` with `status=success`.

Log:
```
[VERIFY] Changed components: frontend/pytorch, common/transformations
```

### Step 2: Build Changed Targets

For each changed component, run a focused build. Use the existing `build/` directory.

```python
import subprocess, sys

build_dir = "build"
build_results = {}

for prefix in changed_components:
    target = target_map.get(prefix)
    if not target:
        continue

    print(f"[VERIFY] Building {target}...")
    result = subprocess.run(
        [sys.executable, "-m", "cmake", "--build", build_dir, "--target", target],
        capture_output=True, text=True,
    )
    # cmake --build is not invoked via python -m; use direct call:
    result = subprocess.run(
        ["cmake", "--build", build_dir, "--target", target],
        capture_output=True, text=True,
    )
    build_results[prefix] = {
        "target": target,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "stderr_tail": result.stderr.strip().splitlines()[-20:] if result.stderr else [],
    }
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[VERIFY] Build {target}: {status}")
```

A build failure is a **hard stop** — do not proceed to unit tests. Write the failure to
`agent-results/verify-implementation/build_results.json` and report `status=failed` to the orchestrator.

### Step 3: Run Unit Tests

For each successfully built component, run its unit tests:

```python
import subprocess

ctest_results = {}

for prefix in changed_components:
    if not build_results.get(prefix, {}).get("ok"):
        continue

    test_target_name = test_target_map.get(prefix)
    if not test_target_name:
        continue

    build_target, ctest_pattern = test_target_name
    print(f"[VERIFY] Running tests: {ctest_pattern}...")

    result = subprocess.run(
        ["ctest", "--test-dir", build_dir, "-R", ctest_pattern,
         "--output-on-failure", "--timeout", "300"],
        capture_output=True, text=True,
    )
    passed = result.returncode == 0
    # Extract summary line (e.g. "100% tests passed, 0 tests failed out of 42")
    summary = next(
        (l for l in result.stdout.splitlines() if "tests passed" in l or "tests failed" in l),
        result.stdout.strip().splitlines()[-1] if result.stdout else "no output"
    )
    ctest_results[prefix] = {
        "returncode": result.returncode,
        "ok": passed,
        "summary": summary,
    }
    status = "PASS" if passed else "FAIL"
    print(f"[VERIFY] Tests {ctest_pattern}: {status} — {summary}")
```

**Tests must be "aggressive"** — do not accept a passing run that skips the new
functionality. Verify that test names covering the newly changed op or pass are
actually executed. If you see `No tests were found` for a newly added test file,
that is a failure — the test was not registered.

### Step 4: Inference Sanity Check

If `agent-results/analyze-and-convert/ov_model_*/openvino_model.xml` exists, run a quick
inference through the locally built OpenVINO to catch segfaults and memory issues:

```python
import glob, pathlib

ir_candidates = sorted(glob.glob(
    "agent-results/analyze-and-convert/ov_model_*/openvino_model.xml"
))

inference_result = {"skipped": True, "reason": "No IR model available"}

if ir_candidates:
    ir_path = ir_candidates[0]
    print(f"[VERIFY] Running inference sanity check on {ir_path}...")
    result = subprocess.run(
        ["python", ".github/scripts/meat/quick_inference_check.py", "--ir", ir_path],
        capture_output=True, text=True, timeout=120,
    )
    inference_result = {
        "skipped": False,
        "ir_path": ir_path,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "output": result.stdout.strip().splitlines()[-10:],
        "stderr_tail": result.stderr.strip().splitlines()[-10:] if result.stderr else [],
    }
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"[VERIFY] Inference check: {status}")
else:
    print("[VERIFY] No IR model found — skipping inference check")
```

The inference check catches:
- Segfaults from incorrect pointer arithmetic in new CPU/GPU kernels
- Out-of-bounds memory access (reports non-zero exit or signal)
- Wrong output shape or NaN/Inf tensors (checked in `quick_inference_check.py`)

A failed inference check is a hard stop even if unit tests pass.

### Step 5: Write Report and Return

```python
import json, datetime

overall_ok = (
    all(r["ok"] for r in build_results.values()) and
    all(r["ok"] for r in ctest_results.values()) and
    (inference_result.get("skipped") or inference_result.get("ok", False))
)

report = {
    "status": "success" if overall_ok else "failed",
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "changed_components": sorted(changed_components),
    "build_results": build_results,
    "test_results": ctest_results,
    "inference_result": inference_result,
}

os.makedirs("agent-results/verify-implementation", exist_ok=True)
with open("agent-results/verify-implementation/verify_result.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"[VERIFY] Overall: {'PASS' if overall_ok else 'FAIL'}")
print(json.dumps(report, indent=2))
```

Log:
```
[VERIFY] build=ok tests=ok inference=ok → overall=PASS
```

---

## Output Contract

| Output field | Type | Description |
|---|---|---|
| `status` | `success` \| `failed` | Overall verification result |
| `changed_components` | list | Components that were verified |
| `build_results` | object | Per-component build outcome |
| `test_results` | object | Per-component test outcome with summary |
| `inference_result` | object | Inference sanity check result (or `skipped: true`) |

Output file: `agent-results/verify-implementation/verify_result.json`

---

## Failure Handling

| Failure type | Action |
|---|---|
| Build fails | Report `status=failed`, include compiler error tail. **Do not run tests.** |
| Unit tests fail | Report `status=failed`, include ctest summary. **Do not run inference check.** |
| Test not found (0 tests executed) | Report `status=failed` — test was not registered correctly. |
| Inference segfault / non-zero exit | Report `status=failed` — this is a kernel or memory bug. |
| NaN/Inf in inference output | Report `status=failed` — this is a precision or initialization bug. |
| IR model not available | Report inference as `skipped` (not failed) — no IR to test against. |

---

## E2E Gate Mode (`mode: e2e_gate`)

When called with `mode: e2e_gate`, run Steps 1–5 (standard build + test verification) first,
then continue with these additional steps.

### Step 6: E2E Conversion Verification

Follow the **[`skills/verify-conversion/SKILL.md`](skills/verify-conversion/SKILL.md)** skill.

The skill:
1. Auto-detects the correct conversion path (optimum-intel for HuggingFace models,
   `ovc`/`convert_model` for local ONNX/PyTorch/TF).
2. Runs one E2E inference through the OV plugin layer.
3. Validates output sanity: no NaN/Inf, non-empty tensors, non-blank LM output.

Writes `agent-results/verify-implementation/e2e_verify.json`.

A failed E2E conversion is a **hard stop** — do not proceed to the checklist.

### Step 7: New-Architecture Validation Checklist

Follow the **[`skills/new-arch-validation-checklist.md`](skills/new-arch-validation-checklist.md)** skill.

For each item, examine changed files (from `agent-results/*/files_created`) and mark
`PASS`, `FAIL`, or `N/A`. Log the result per category.

Writes `agent-results/verify-implementation/checklist_result.json`.

### Step 8: Sub-Agent Test Result Scan

```
python .github/scripts/meat/check_subagent_results.py
```

The script scans all sub-agent result JSONs for `status=failed` or failing `test_results`.
Exits with code 1 and prints details if any failure is found.

### E2E Gate Output

Write combined result to `agent-results/verify-implementation/e2e_result.json`:
```json
{
  "status": "success|failed",
  "build_and_tests": "<path to verify_result.json>",
  "e2e_verify": {"ok": true, "path": "e2e_verify.json"},
  "checklist": {"pass": 12, "fail": 0, "na": 3},
  "subagent_tests": "pass|fail"
}
```

| Condition | `e2e_result.status` |
|---|---|
| Build + tests + E2E + checklist + sub-agent scan all pass | `success` |
| Build or unit tests fail | `failed` (from Step 2/3; Steps 6–8 are not run) |
| E2E conversion fails | `failed` |
| Checklist has any `FAIL` | `failed` |
| Sub-agent scan exits non-zero | `failed` |

---

## Constraints

- Does **not** modify any source code.
- Does **not** create PRs or branches.
- Reports only to Enable Operator Agent (OV Orchestrator).
- If `build/` does not exist or CMake was not configured, report `status=failed` with reason
  `"build directory not found"` — do not attempt to configure CMake.

---
name: build-report
description: Combine all gathered artifacts into a structured conversion_report.md and a machine-readable agent-complete marker. Posts the report to the GitHub issue and writes the manifest entry.
---

# Skill: Build Report

**Trigger:** Final step of the `analyze-and-convert` workflow. Always runs —
regardless of whether conversion succeeded or failed.

Inputs: `model_profile.json`, `conversion_attempts.json`, `routing_signals.json`,
`error_excerpts.json` (all produced by previous skills).

---

## Step 1 — Load all artifacts

```python
import json, os

def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default or {}

profile   = load_json("model_profile.json")
attempts  = load_json("conversion_attempts.json", default=[])
signals   = load_json("routing_signals.json")
excerpts  = load_json("error_excerpts.json", default={})

MODEL_ID  = profile.get("model_id", "unknown")
SUCCEEDED = any(a.get("success") for a in attempts)
winning   = next((a for a in attempts if a.get("success")), None)
```

---

## Step 2 — Write conversion_report.md

```python
lines = []
status_icon = "[OK]" if SUCCEEDED else "[FAIL]"
status_text = "SUCCEEDED" if SUCCEEDED else "FAILED"

lines += [
    f"# Conversion Report: {MODEL_ID}",
    "",
    f"**Status:** {status_icon} {status_text}",
    "",
    "## Model Profile",
    "",
    f"| Property | Value |",
    f"|----------|-------|",
    f"| Model ID | `{profile.get('model_id')}` |",
    f"| Model type | `{profile.get('model_type')}` |",
    f"| Architectures | {profile.get('architectures')} |",
    f"| Pipeline tag | `{profile.get('pipeline_tag')}` |",
    f"| Estimated params | ~{profile.get('estimated_params_b', '?')}B |",
    f"| VLM | {'Yes' if profile.get('is_vlm') else 'No'} |",
    f"| trust_remote_code required | {'Yes' if profile.get('trust_remote_code_required') else 'No'} |",
    f"| optimum-intel registered | {'Yes' if profile.get('optimum_supported') else '**No** — new arch needed'} |",
    "",
]

# Conversion attempts table
lines += [
    "## Conversion Attempts",
    "",
    "| Strategy | Description | Result | Time |",
    "|----------|-------------|--------|------|",
]
for a in attempts:
    result = "Success" if a.get("success") else f"FAIL rc={a.get('returncode', '?')}"
    lines.append(
        f"| `{a['id']}` | {a['description']} | {result} | {a.get('elapsed_s', '?')}s |"
    )
lines.append("")

# Winning strategy detail
if winning:
    lines += [
        "## Successful Conversion",
        "",
        f"**Strategy:** `{winning['id']}` — {winning['description']}",
        f"**Command:**",
        "```bash",
        winning.get("command", ""),
        "```",
        f"**IR files:** {', '.join(winning.get('ir_files', []))}",
    ]
    if winning.get("inference_ok"):
        lines += [
            "",
            f"**Inference check:** Passed",
            f"**Sample output:** `{winning.get('inference_sample', '')}`",
        ]
    elif "inference_ok" in winning:
        lines += [
            "",
            f"**Inference check:** Failed",
            "```",
            winning.get("inference_error", ""),
            "```",
        ]
    lines.append("")

# Failure details
if not SUCCEEDED:
    lines += ["## Failure Details", ""]
    for a in attempts:
        if not a.get("success"):
            excerpt = excerpts.get(a["id"], "(no traceback captured)")
            lines += [
                f"### Strategy `{a['id']}`",
                "",
                f"**Command:** `{a.get('command', '')}`",
                "",
                "**Error excerpt:**",
                "```",
                excerpt,
                "```",
                "",
            ]

# Routing signals
lines += [
    "## Routing Signals",
    "",
    f"| Signal | Value |",
    f"|--------|-------|",
]
for k, v in signals.items():
    lines.append(f"| `{k}` | `{v}` |")

lines += [
    "",
    "## Recommended Next Step",
    "",
]

if SUCCEEDED and winning.get("inference_ok"):
    lines += [
        "Model exported and inference verified. Route to **WWB** for accuracy benchmark.",
        "",
        f"- `next_agent: wwb`",
        f"- `status: success`",
        f"- IR directory: `{winning.get('ir_dir', 'ov_model')}`",
    ]
elif SUCCEEDED and not winning.get("inference_ok", True):
    lines += [
        "Export succeeded but inference check failed.",
        f"Route to **{signals.get('target_agent', 'openvino-orchestrator')}** for inference debugging.",
        "",
        f"- `next_agent: {signals.get('target_agent')}`",
        f"- `status: partial`",
        f"- `error_class: inference_runtime_error`",
    ]
else:
    ec = signals.get("error_class", "unknown")
    ta = signals.get("target_agent", "optimum-intel")
    notes = []
    if signals.get("requires_optimum_new_arch"):
        notes.append("Architecture not in optimum-intel — full new-arch workflow needed.")
    if signals.get("requires_transformers_upgrade"):
        notes.append(f"transformers override: `{signals.get('transformers_override')}`")
    if signals.get("custom_ops_suspected"):
        notes.append("Custom ops suspected (SSM/recurrent) — `_ov_ops.py` likely needed.")
    if signals.get("oom_suspected"):
        notes.append("OOM detected — consider int4 quantization or sharded export.")
    if signals.get("is_vlm"):
        notes.append("VLM model — tokenizer check required after optimum-intel fix.")

    lines += [
        f"All {len(attempts)} conversion strategies failed.",
        f"Route to **`{ta}`** with error class `{ec}`.",
        "",
        f"- `next_agent: {ta}`",
        f"- `status: failed`",
        f"- `error_class: {ec}`",
    ]
    if notes:
        lines += ["", "**Notes:**"] + [f"  - {n}" for n in notes]

report_text = "\n".join(lines)
with open("conversion_report.md", "w", encoding="utf-8") as f:
    f.write(report_text)
print("Written: conversion_report.md")
```

---

## Step 3 — Post to GitHub Issue (if GITHUB_TOKEN available)

```python
import os, subprocess, sys

issue_number = os.environ.get("ISSUE_NUMBER")
github_token = os.environ.get("GITHUB_TOKEN")
repo         = os.environ.get("GITHUB_REPOSITORY")

if issue_number and github_token and repo:
    # Post via scripts/post_issue_comment.py
    result = subprocess.run(
        [sys.executable, "scripts/post_issue_comment.py",
         "--issue", issue_number,
         "--body-file", "conversion_report.md"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Posted conversion_report.md to GitHub issue.")
    else:
        print(f"GitHub post failed: {result.stderr}")
else:
    print("GITHUB_TOKEN / ISSUE_NUMBER not set — skipping GitHub post.")
```

---

## Step 4 — Write Manifest Entry

```python
import os, subprocess, sys

agent_name = "analyze-and-convert"
run_id = os.environ.get("GITHUB_RUN_ID", "local")
pass_num = os.environ.get("PASS_NUM", "1")

entry_type = "model_ir" if SUCCEEDED else "analysis"
description = (
    f"Conversion succeeded via {winning['id']}" if SUCCEEDED
    else f"{signals.get('error_class')} — target: {signals.get('target_agent')}"
)

subprocess.run([
    sys.executable, "scripts/collect_artifacts.py", "add",
    "--agent", agent_name,
    "--pass", pass_num,
    "--type", entry_type,
    "--component", "openvino",
    "--artifact-name", f"conversion-report-{run_id}",
    "--description", description,
], check=False)
```

---

## Step 5 — Write agent-complete Marker

Print the standard marker for `dispatcher.yml` to detect and route the pipeline.

```python
import json

status = "success" if (SUCCEEDED and winning.get("inference_ok")) else (
    "partial" if SUCCEEDED else "failed"
)
next_agent = "wwb" if status == "success" else signals.get("target_agent", "optimum-intel")
error_class = "none" if status == "success" else signals.get("error_class", "unknown")

# Build next_context — all signals the next agent needs
next_context = {k: v for k, v in signals.items()
                if k not in ("target_agent",) and v not in (False, "", None)}

print("\n<!-- agent-complete")
print(json.dumps({
    "agent": "analyze-and-convert",
    "status": status,
    "next_agent": next_agent,
    "error_class": error_class,
    "next_context": json.dumps(next_context),
    "model_id": MODEL_ID,
}, indent=2))
print("-->")
```

---

## Output

| File | Contents |
|------|----------|
| `conversion_report.md` | Full human-readable report — posted to GitHub issue |
| `agent-complete` marker (stdout) | Machine-readable routing block for `dispatcher.yml` |

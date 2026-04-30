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
        f"Route to **{signals.get('target_agent', 'enable-operator')}** for inference debugging.",
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
# Optionally post as PR comment if gh CLI is available
import shutil
if shutil.which("gh") and github_token:
    import subprocess
    result = subprocess.run(
        ["gh", "pr", "comment", "--body-file", "conversion_report.md"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("Posted conversion_report.md as PR comment.")
    else:
        print(f"gh pr comment failed (non-fatal): {result.stderr}")
else:
    print("gh not available or GITHUB_TOKEN not set — skipping PR comment (report saved to file).")
```

---

## Step 4 — Record Result

```python
import json

agent_name = "analyze-and-convert"

entry_type = "model_ir" if SUCCEEDED else "analysis"
description = (
    f"Conversion succeeded via {winning['id']}" if SUCCEEDED
    else f"{signals.get('error_class')} — target: {signals.get('target_agent')}"
)

# Record result summary in agent-results/ for orchestrator pickup
result_summary = {
    "agent": agent_name,
    "status": status,
    "model_id": MODEL_ID,
    "entry_type": entry_type,
    "description": description,
    "report_path": "conversion_report.md",
}
with open("analyze_and_convert_result.json", "w") as f:
    json.dump(result_summary, f, indent=2)
print("Written: analyze_and_convert_result.json")
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
| `conversion_report.md` | Full human-readable report — saved to agent-results/ |
| `agent-complete` marker (stdout) | Machine-readable routing block for `dispatcher.yml` |

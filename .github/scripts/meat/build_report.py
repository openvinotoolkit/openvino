# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Build-report step for the analyze-and-convert agent.

Loads all artifacts produced by previous skills and:
  1. Generates conversion_report.md (human-readable)
  2. Posts the report as a PR/issue comment via gh CLI (if available)
  3. Writes analyze_and_convert_result.json (machine-readable)
  4. Prints the <!-- agent-complete --> marker on stdout

Usage:
    python .github/scripts/meat/build_report.py

Expected input files (all in the current working directory):
    model_profile.json
    conversion_attempts.json
    routing_signals.json
    error_excerpts.json

Output files:
    agent-results/analyze-and-convert/conversion_report.md
    agent-results/analyze-and-convert/analyze_and_convert_result.json
"""

import json
import os
import pathlib
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = pathlib.Path("agent-results") / "analyze-and-convert"

INPUTS = {
    "profile":   pathlib.Path("model_profile.json"),
    "attempts":  pathlib.Path("conversion_attempts.json"),
    "signals":   pathlib.Path("routing_signals.json"),
    "excerpts":  pathlib.Path("error_excerpts.json"),
}

REPORT_PATH = RESULTS_DIR / "conversion_report.md"
RESULT_JSON = RESULTS_DIR / "analyze_and_convert_result.json"


# ---------------------------------------------------------------------------
# Step 1 — Load artifacts
# ---------------------------------------------------------------------------

def load_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[build_report] WARNING: could not parse {path}: {exc}", file=sys.stderr)
    else:
        print(f"[build_report] WARNING: {path} not found — using default", file=sys.stderr)
    return default


profile  = load_json(INPUTS["profile"],  {})
attempts = load_json(INPUTS["attempts"], [])
signals  = load_json(INPUTS["signals"],  {})
excerpts = load_json(INPUTS["excerpts"], {})

model_id     = profile.get("model_id", "unknown")
error_class  = signals.get("error_class", "unknown")
target_agent = signals.get("target_agent", "optimum-intel")

# Determine overall status
successful = [a for a in attempts if a.get("success")]
if not successful:
    status = "failed"
else:
    winning = successful[-1]
    if winning.get("inference_ok") is True:
        status = "success"
    elif "inference_ok" in winning:
        # Exported OK but inference check failed
        status = "partial"
    else:
        status = "success"


# ---------------------------------------------------------------------------
# Step 2 — Write conversion_report.md
# ---------------------------------------------------------------------------

def md_table(headers: list, rows: list) -> str:
    sep = " | ".join("---" for _ in headers)
    header_row = " | ".join(headers)
    lines = [f"| {header_row} |", f"| {sep} |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def build_report() -> str:
    lines = []

    lines.append(f"# Conversion Report: `{model_id}`\n")
    lines.append(f"**Overall status:** `{status}`  \n")
    lines.append(f"**Error class:** `{error_class}`  \n")
    lines.append(f"**Recommended next agent:** `{target_agent}`\n")

    # --- Model profile ---
    lines.append("\n## Model Profile\n")
    if profile:
        profile_rows = [
            [k, str(v)]
            for k, v in profile.items()
            if k != "special_config_keys"
        ]
        lines.append(md_table(["Key", "Value"], profile_rows))
        special_keys = profile.get("special_config_keys", [])
        if special_keys:
            lines.append(f"\n**Special config keys:** {', '.join(special_keys)}")
    else:
        lines.append("_No profile data available._")

    # --- Conversion attempts ---
    lines.append("\n\n## Conversion Attempts\n")
    if attempts:
        attempt_rows = [
            [
                a.get("id", "?"),
                "✅" if a.get("success") else "❌",
                a.get("description", ""),
                a.get("weight_format", ""),
                a.get("optimum_version", ""),
            ]
            for a in attempts
        ]
        lines.append(md_table(
            ["ID", "Result", "Description", "Weight Format", "Optimum Version"],
            attempt_rows,
        ))
    else:
        lines.append("_No conversion attempts recorded._")

    # --- Successful strategy ---
    if successful:
        lines.append("\n\n## Successful Strategy\n")
        s = successful[-1]
        lines.append(f"Strategy **{s.get('id')}** succeeded.\n")
        extra = {k: v for k, v in s.items()
                 if k not in ("id", "success", "description", "weight_format",
                              "optimum_version", "stdout", "stderr")}
        if extra:
            lines.append(md_table(["Flag", "Value"], list(extra.items())))

    # --- Failure details ---
    if not successful:
        lines.append("\n\n## Failure Details\n")
        for a in attempts:
            if not a.get("success"):
                lines.append(f"### Attempt `{a.get('id', '?')}`\n")
                excerpt = excerpts.get(a.get("id", ""), "")
                if excerpt:
                    lines.append("```")
                    lines.append(textwrap.indent(excerpt, "  "))
                    lines.append("```\n")
                else:
                    lines.append("_No error excerpt available._\n")

    # --- Routing signals ---
    lines.append("\n## Routing Signals\n")
    if signals:
        signal_rows = [[k, str(v)] for k, v in signals.items()]
        lines.append(md_table(["Signal", "Value"], signal_rows))
    else:
        lines.append("_No routing signals available._")

    # --- Recommended next step ---
    lines.append("\n\n## Recommended Next Step\n")
    _next_step_map = {
        "optimum-intel":       "Open a follow-up issue against **Optimum-Intel**.",
        "enable-operator":     "Dispatch to **OV Orchestrator** (`enable-operator` agent).",
        "openvino-genai":      "Dispatch to **GenAI** team for pipeline support.",
        "openvino-tokenizers": "Dispatch to **Tokenizers** team for tokenizer conversion fix.",
    }
    lines.append(_next_step_map.get(target_agent, f"Route to `{target_agent}`."))

    return "\n".join(lines) + "\n"


RESULTS_DIR.mkdir(parents=True, exist_ok=True)
report_text = build_report()
REPORT_PATH.write_text(report_text, encoding="utf-8")
print(f"[build_report] Wrote {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Step 3 — Post to GitHub PR/Issue (best-effort)
# ---------------------------------------------------------------------------

def post_to_github(report_path: pathlib.Path) -> None:
    pr_number = os.environ.get("PR_NUMBER", "")
    issue_number = os.environ.get("ISSUE_NUMBER", "")
    if not (pr_number or issue_number):
        print("[build_report] No PR_NUMBER or ISSUE_NUMBER set — skipping gh comment")
        return

    try:
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("[build_report] gh CLI not available — skipping comment post")
        return

    body = report_path.read_text(encoding="utf-8")
    if pr_number:
        cmd = ["gh", "pr", "comment", pr_number, "--body", body]
    else:
        cmd = ["gh", "issue", "comment", issue_number, "--body", body]

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        print("[build_report] Posted report to GitHub")
    else:
        print(f"[build_report] gh comment failed: {result.stderr.strip()}", file=sys.stderr)


post_to_github(REPORT_PATH)


# ---------------------------------------------------------------------------
# Step 4 — Write analyze_and_convert_result.json
# ---------------------------------------------------------------------------

result_data = {
    "agent": "analyze-and-convert",
    "status": status,
    "model_id": model_id,
    "entry_type": profile.get("pipeline_tag", "unknown"),
    "error_class": error_class,
    "target_agent": target_agent,
    "routing_signals": {
        k: signals[k]
        for k in (
            "requires_optimum_new_arch",
            "requires_transformers_upgrade",
            "transformers_override",
            "requires_tokenizer_check",
            "trust_remote_code_required",
            "is_vlm",
            "custom_ops_suspected",
            "oom_suspected",
        )
        if k in signals
    },
}

RESULT_JSON.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
print(f"[build_report] Wrote {RESULT_JSON}")


# ---------------------------------------------------------------------------
# Step 5 — Print agent-complete marker
# ---------------------------------------------------------------------------

next_context = {
    k: signals[k]
    for k in (
        "requires_optimum_new_arch",
        "requires_transformers_upgrade",
        "transformers_override",
        "requires_tokenizer_check",
        "trust_remote_code_required",
        "is_vlm",
        "custom_ops_suspected",
        "oom_suspected",
    )
    if k in signals
}

marker = {
    "agent": "analyze-and-convert",
    "status": status,
    "next_agent": target_agent,
    "error_class": error_class,
    "model_id": model_id,
    "next_context": next_context,
}

print("<!-- agent-complete")
print(json.dumps(marker, indent=2))
print("-->")

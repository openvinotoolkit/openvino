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

Run `build_report.py` to load artifacts and compute status:

```bash
python build_report.py
```

Script: [build_report.py](./build_report.py)

---

## Step 2 — Write conversion_report.md

Script: [build_report.py](./build_report.py) — generates `conversion_report.md` with:

- Model profile table
- Conversion attempts summary
- Successful strategy details (if applicable)
- Failure details and error excerpts (if applicable)
- Routing signals summary
- Recommended next step and agent routing

---

## Step 3 — Post to GitHub Issue (if gh CLI available)

Script: [build_report.py](./build_report.py) — posts `conversion_report.md` as a PR comment using `gh pr comment` if available.

---

## Step 4 — Record Result

Script: [build_report.py](./build_report.py) — writes `analyze_and_convert_result.json` with:

- Agent name and status (success/partial/failed)
- Model ID and entry type
- Error classification and target agent routing

---

## Step 5 — Write agent-complete Marker

Script: [build_report.py](./build_report.py) — prints standard `<!-- agent-complete -->` marker (HTML comment) with:

- `agent`: "analyze-and-convert"
- `status`: "success" | "partial" | "failed"
- `next_agent`: target agent for routing (wwb, optimum-intel, etc.)
- `error_class`: error classification (none, oom_suspected, custom_ops, etc.)
- `next_context`: filtered signals for next agent
- `model_id`: original model identifier

---

## Output

| File | Contents |
|------|----------|
| `conversion_report.md` | Full human-readable report — saved to agent-results/ |
| `agent-complete` marker (stdout) | Machine-readable routing block for `dispatcher.yml` |

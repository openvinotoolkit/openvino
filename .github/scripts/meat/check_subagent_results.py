# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Phase 6 helper for enable-operator agent.

Scans sub-agent result JSON files for failures and failing test results.
Prints a summary and exits with code 1 if any failure is found, 0 otherwise.

Usage:
    python .github/scripts/meat/check_subagent_results.py
"""

import json
import pathlib
import sys

SUBAGENT_RESULTS = [
    "agent-results/frontend/fe_result.json",
    "agent-results/core-opspec/core_opspec_result.json",
    "agent-results/transformation/transformation_result.json",
    "agent-results/cpu/cpu_result.json",
    "agent-results/gpu/gpu_result.json",
]

failures = []

for path_str in SUBAGENT_RESULTS:
    p = pathlib.Path(path_str)
    if not p.exists():
        continue

    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{path_str}: malformed JSON — {exc}")
        continue

    if d.get("status") == "failed":
        failures.append(f"{path_str}: status=failed")

    tr = d.get("test_results", "")
    if tr and "FAILED" in str(tr).upper():
        failures.append(f"{path_str}: test_results={tr!r}")

if failures:
    print("[OV-ORCH] [phase=e2e-gate] SUB-AGENT TEST FAILURES:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)

print("[OV-ORCH] [phase=e2e-gate] All sub-agent results: PASS")

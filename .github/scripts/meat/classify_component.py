# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Phase 1 helper for enable-operator agent.

Reads agent-results/pipeline_state.json, classifies the failing component
from error_context, and prints the result as 'component=<value>' to stdout.

Exit codes:
  0  — classification succeeded (result printed)
  0  — state file missing (prints 'component=frontend' as safe default)
"""

import json
import pathlib
import sys

CLASSIFICATION_MAP = {
    "missing_conversion_rule": "frontend",
    "frontend_error": "frontend",
    "ir_validation_error": "core_op",
    "inference_runtime_error": "cpu_plugin",
    "accuracy_regression": "transformation",
}

STATE = pathlib.Path("agent-results/pipeline_state.json")

if not STATE.exists():
    print("component=frontend")
    sys.exit(0)

d = json.loads(STATE.read_text(encoding="utf-8"))
orch = d.get("ov_orchestrator", {})
error_context = orch.get("error_context", "")

error_class = error_context.split("/")[0].strip() if error_context else "unknown"
component = CLASSIFICATION_MAP.get(error_class, "frontend")

# Multi-op detection: report co_located_ops when present
co_located = orch.get("co_located_ops", [])
if co_located:
    print(f"[OV-ORCH] Multi-op detected: co_located_ops={co_located} — routing as single target")

print(f"component={component}")
print(f"[OV-ORCH] Classified component: {component} (error_class={error_class})")

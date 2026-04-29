# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Phase 7 helper for enable-operator agent — patch collection.

Reads patch paths from sub-agent result JSON files and copies them to
agent-results/enable-operator/patches/openvino/.
Produces a combined patch file for convenience.

Usage:
    python .github/scripts/meat/collect_patches.py

Exit codes:
  0 — patches collected (or zero patches found — no-op)
  1 — at least one referenced patch file is missing
"""

import json
import pathlib
import shutil
import sys

RESULT_FILES = [
    "agent-results/frontend/fe_result.json",
    "agent-results/core-opspec/core_opspec_result.json",
    "agent-results/transformation/transformation_result.json",
    "agent-results/cpu/cpu_result.json",
    "agent-results/gpu/gpu_result.json",
    "agent-results/npu/npu_result.json",
]

OUT_DIR = pathlib.Path("agent-results/enable-operator/patches/openvino")
OUT_DIR.mkdir(parents=True, exist_ok=True)

collected: list[pathlib.Path] = []
missing: list[str] = []

for path_str in RESULT_FILES:
    p = pathlib.Path(path_str)
    if not p.exists():
        continue

    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue

    patch_paths = d.get("patch_paths", [])
    if not patch_paths and d.get("patch_path"):
        patch_paths = [d["patch_path"]]

    for pp in patch_paths:
        if not pp:
            continue
        src = pathlib.Path(pp)
        if src.is_file():
            dst = OUT_DIR / src.name
            shutil.copy(src, dst)
            collected.append(dst)
        else:
            missing.append(str(pp))
            print(f"[WARN] Patch not found: {pp} — skipping", file=sys.stderr)

if not collected:
    print("[WARN] No patches collected — nothing to publish")
    sys.exit(0)

# Combine all patches into a single file (useful for review)
combined = pathlib.Path("agent-results/enable-operator/patches/openvino_combined.patch")
combined.write_text(
    "\n".join(c.read_text(encoding="utf-8") for c in collected),
    encoding="utf-8",
)
print(f"[OV-ORCH] Combined {len(collected)} patches into {combined}")

if missing:
    sys.exit(1)

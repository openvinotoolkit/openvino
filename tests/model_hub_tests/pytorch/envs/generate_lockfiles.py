#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Generate lockfiles for PyTorch model hub test environments.

Each lockfile is a fully-resolved set of packages produced by `uv pip compile`.
`uv pip sync <lockfile>` will then make the environment match it exactly —
installing missing packages AND removing extras.

Usage:
    python3 generate_lockfiles.py [--python-version 3.11]

Lockfiles are written to tests/model_hub_tests/pytorch/envs/locks/.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TESTS_DIR = SCRIPT_DIR.parent.parent.parent  # tests/
ENVS_DIR = SCRIPT_DIR  # tests/model_hub_tests/pytorch/envs/
LOCKS_DIR = ENVS_DIR / "locks"
BASE_REQS = TESTS_DIR / "requirements_pytorch.txt"

# Each group maps to: (lockfile_name, {"reqs": [...], "overrides": [...]})
#   reqs:      overlay files with additional packages (merged with base)
#   overrides: overlay files that replace base pins (e.g. different torch version)
# Every lockfile includes the base requirements + the listed overlays.
GROUPS = {
    "base": {"reqs": [], "overrides": []},
    "timm_torchvision": {"reqs": ["timm.txt", "edsr.txt", "torchvision.txt"], "overrides": []},
    "hf_models": {"reqs": ["hf_transformers.txt", "easyocr.txt", "gfpgan.txt", "tpsmm.txt"], "overrides": []},
    "llm": {"reqs": ["hf_transformers.txt", "llm.txt"], "overrides": []},
    "compile_gptq": {"reqs": ["hf_transformers.txt", "llm.txt"], "overrides": ["compile_gptq.txt"]},
    "moe": {"reqs": ["hf_transformers.txt", "llm.txt"], "overrides": ["moe.txt"]},
}


def compile_lockfile(group_name: str, config: dict, python_version: str) -> None:
    """Run uv pip compile to produce a lockfile for the given group."""
    lock_path = LOCKS_DIR / f"{group_name}.lock"

    cmd = [
        sys.executable, "-m", "uv", "pip", "compile",
        str(BASE_REQS),
        "--python-version", python_version,
        "--output-file", str(lock_path),
        "--no-header",
        "--no-annotate",
        # PyTorch's index is listed via --extra-index-url in base requirements.
        # Without this flag, uv refuses to pick packages from PyPI when they
        # also appear on the PyTorch index at a different version.
        "--index-strategy", "unsafe-best-match",
    ]

    for overlay in config["reqs"]:
        overlay_path = ENVS_DIR / overlay
        if not overlay_path.exists():
            print(f"  WARNING: {overlay_path} not found, skipping")
            continue
        cmd.append(str(overlay_path))

    for override in config["overrides"]:
        override_path = ENVS_DIR / override
        if not override_path.exists():
            print(f"  WARNING: {override_path} not found, skipping")
            continue
        cmd.extend(["--overrides", str(override_path)])

    print(f"  Generating {lock_path.name} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR compiling {group_name}:\n{result.stderr}")
        sys.exit(1)
    else:
        # Strip local version tags like +cpu / +cu121 so lockfiles are
        # platform-agnostic.  The correct wheel variant is picked at
        # install time via --extra-index-url.
        raw = lock_path.read_text()
        cleaned = re.sub(r"(\S)==(\S+)\+\w+", r"\1==\2", raw)
        lock_path.write_text(cleaned)

        n_lines = sum(1 for line in cleaned.splitlines() if line and not line.startswith("#"))
        print(f"  OK — {n_lines} packages locked")


def main():
    parser = argparse.ArgumentParser(description="Generate PyTorch test env lockfiles")
    parser.add_argument("--python-version", default="3.11",
                        help="Target Python version for resolution (default: 3.11)")
    args = parser.parse_args()

    LOCKS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Base requirements: {BASE_REQS}")
    print(f"Output directory:  {LOCKS_DIR}")
    print(f"Python version:    {args.python_version}")
    print()

    for group_name, config in GROUPS.items():
        compile_lockfile(group_name, config, args.python_version)

    print()
    print("Done. Lockfiles:")
    for f in sorted(LOCKS_DIR.glob("*.lock")):
        print(f"  {f.relative_to(TESTS_DIR)}")


if __name__ == "__main__":
    main()

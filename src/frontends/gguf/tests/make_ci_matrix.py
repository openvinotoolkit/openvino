# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Prepare the Linux x64 GGUF C++ test matrix from an installed OpenVINO artifact."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install = args.install_dir.resolve()
    tests = install / "tests"
    info = tests / "gguf_validation/ci_build_info.json"
    context = json.loads(info.read_text())
    if not context.get("artifact_revision"):
        parser.error("The test artifact must record its build revision")
    context["coverage"] = "Linux x64, full frontend and architecture-library suites with generated architecture fixtures"
    binaries = [("frontend", "ov_gguf_frontend_tests", "TEST-GGUFFrontend.xml", 511, 1),
                ("architecture-library", "ov_gguf_architecture_library_tests", "TEST-GGUFArchitectureLibrary.xml", 1, 0)]
    cases = []
    for name, binary, report, count, skips in binaries:
        executable, xml = tests / binary, tests / report
        if not executable.is_file():
            parser.error(f"Missing test executable: {executable}")
        cases.append({"id": name, "cwd": str(tests),
                      "command": [str(executable), "--gtest_print_time=1", f"--gtest_output=xml:{xml}"],
                      "env": {"OV_GGUF_Q4_K_ZP_F16": "0"}, "timeout_seconds": 1800,
                      "metadata": {"acceptance": "All tests pass; only the optional real embedding test may skip"
                                   if skips else "All tests pass without skips"},
                      "reports": [{"kind": "gtest", "path": str(xml), "expected_tests": count, "max_skips": skips}]})
    inputs = {info, Path(__file__).resolve(), *(tests / binary for _, binary, _, _, _ in binaries)}
    inputs.update(p for p in (tests / "test_data").rglob("*")
                  if p.is_file() and (p.suffix in {".npy", ".npz", ".hdr"} or p.name == "manifest.txt"))
    inputs.update(p for p in install.rglob("*") if p.is_file() and ".so" in p.name)
    manifest = {"version": 1, "context": context, "repositories": [],
                "inputs": sorted(str(p) for p in inputs), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

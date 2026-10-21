#!/usr/bin/env python3
#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Generate bytecode compatibility artifacts from YAML configurations.

Usage:
python3 tests/artifacts/bytecode/generate_bytecode_artifact.py \
    tests/artifacts/bytecode/configs/*.yaml \
    --output-dir tests/artifacts/bytecode/binary/

The artifact model is split across the `bytecode_gen` package:
    - encoders.py          little-endian byte encoders
    - model.py             section/opcode/type enums and dataclasses
    - function_builder.py  per-function instruction stream builder
    - artifact_builder.py  binary artifact assembly
    - config.py            YAML parsing and config-driven generation
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bytecode_gen.config import build_from_config, load_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate bytecode compatibility artifacts from YAML configurations")
    parser.add_argument("configs", nargs="+", help="One or more YAML config files")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for generated binaries")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for config_file in args.configs:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Warning: config file not found: {config_file}")
            continue

        config = load_yaml(config_path)

        # Prefer the config's explicit 'output' name; otherwise derive it from the
        # config filename (foo.yaml -> foo.bin).
        output_filename = config.get("output")
        if output_filename is None:
            output_filename = config_path.with_suffix(".bin").name

        output_path = output_dir / output_filename
        out = build_from_config(config, output_path=output_path)
        print(f"Generated artifact: {out}")


if __name__ == "__main__":
    main()

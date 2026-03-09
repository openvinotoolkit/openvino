#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Append CI environment variables to GITHUB_ENV.

Usage:
    python3 append_variables.py --runner-os <Linux|Windows|macOS> \
        --linux-path <path> --windows-path <path>

The script reads a secret token from a file, masks it in GitHub Actions logs,
and appends it as HF_TOKEN to the GITHUB_ENV file so subsequent steps can use it.
"""

import argparse
import os
import sys


def load_token_linux(path: str) -> str:
    """Load a secret token from a file path on Linux/macOS runners."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Token file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_token_windows(path: str) -> str:
    """Load a secret token from a file path on Windows runners."""
    if not os.path.exists(path):
        print(f"⚠ HuggingFace token file not found at {path}, skipping", file=sys.stderr)
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def append_to_github_env(key: str, value: str) -> None:
    """Write a key=value pair to the GITHUB_ENV file consumed by subsequent steps."""
    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        print("Warning: GITHUB_ENV is not set; variable will not persist.", file=sys.stderr)
        return
    with open(github_env, "a", encoding="utf-8") as f:
        f.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append environment variables from secret files to GITHUB_ENV."
    )
    parser.add_argument(
        "--runner-os",
        required=True,
        choices=["Linux", "Windows", "macOS"],
        help="The runner operating system (value of ${{ runner.os }}).",
    )
    parser.add_argument(
        "--linux-path",
        default=None,
        help="Absolute path to the HF token file on Linux/macOS runners.",
    )
    parser.add_argument(
        "--windows-path",
        default=None,
        help="Absolute path to the HF token file on Windows runners.",
    )
    args = parser.parse_args()

    if args.runner_os == "Windows":
        if not args.windows_path:
            print("⚠ --windows-path not provided, skipping HF_TOKEN setup.", file=sys.stderr)
            return
        token = load_token_windows(args.windows_path)
        if not token:
            return
    else:
        if not args.linux_path:
            print("⚠ --linux-path not provided, skipping HF_TOKEN setup.", file=sys.stderr)
            return
        token = load_token_linux(args.linux_path)

    # Mask the token value in all subsequent GitHub Actions log output.
    print(f"::add-mask::{token}")

    append_to_github_env("HF_TOKEN", token)
    print("✓ HuggingFace token loaded and masked")


if __name__ == "__main__":
    main()

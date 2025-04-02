# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script is used in the CI to compare the generated .pyi files with the committed .pyi files

import os
import sys
import filecmp
from typing import List, Set


def find_pyi_files(directory: str) -> Set[str]:
    """Recursively find all .pyi files in a directory."""
    pyi_files: List[str] = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pyi"):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                pyi_files.append(relative_path)
    return set(pyi_files)


def compare_pyi_files(generated_dir: str, committed_dir: str) -> None:
    """Compare .pyi files between two directories."""
    # Find all .pyi files in both directories
    generated_files: Set[str] = find_pyi_files(generated_dir)
    committed_files: Set[str] = find_pyi_files(committed_dir)

    # Assert that the number of .pyi files matches
    if len(generated_files) != len(committed_files):
        print(f"Error: Number of .pyi files does not match. "
              f"Generated: {len(generated_files)}, Committed: {len(committed_files)}")
        sys.exit(1)

    # Assert that each file has a pair
    if generated_files != committed_files:
        missing_in_generated = committed_files - generated_files
        missing_in_committed = generated_files - committed_files
        if missing_in_generated:
            print(f"Error: Stub file missing in generated directory: {missing_in_generated}")
        if missing_in_committed:
            print(f"Error: Stub file missing in committed directory: {missing_in_committed}")
        sys.exit(1)

    # Assert that the files are identical
    for relative_path in generated_files:
        generated_file: str = os.path.join(generated_dir, relative_path)
        committed_file: str = os.path.join(committed_dir, relative_path)
        if not filecmp.cmp(generated_file, committed_file, shallow=False):
            print(f"Error: Stub files differ: {relative_path}")
            sys.exit(1)

    print("All .pyi files match.")
    sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_pyi_files.py <generated_dir> <committed_dir>")
        sys.exit(1)

    generated_dir: str = sys.argv[1]
    committed_dir: str = sys.argv[2]

    if not os.path.isdir(generated_dir):
        print(f"Error: Generated directory does not exist: {generated_dir}")
        sys.exit(1)

    if not os.path.isdir(committed_dir):
        print(f"Error: Committed directory does not exist: {committed_dir}")
        sys.exit(1)

    compare_pyi_files(generated_dir, committed_dir)

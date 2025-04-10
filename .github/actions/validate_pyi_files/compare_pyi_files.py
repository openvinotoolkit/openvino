# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This script is used in the CI to compare the generated .pyi files with the committed .pyi files

import os
import sys
import filecmp
import difflib
from typing import List, Set

# Due to numerous issues with stub generation reproducibility we need some skips.
# The inconsistencies between generations most likely come from different environment/CMake setups.
SKIPS = [
    # This file contains constant.def_buffer(), which pybind11-stubgen generates methods for on CI
    # but not locally (__buffer__(), __release_buffer__())
    "openvino/_pyopenvino/op/__init__.pyi",

    # The IOType class sometimes has a docstring, breaking the CI check
    "openvino/tools/ovc/moc_frontend/extractor.pyi"
]

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
        # TF FE stubs can be generated on not depending on the CMake flags in CI
        committed_files = {file for file in committed_files if not file.startswith("openvino/frontend/tensorflow")}

        # Check again after removal
        if len(generated_files) != len(committed_files):
            print(f"Error: Number of .pyi files does not match. "
                  f"Generated (reference): {len(generated_files)}, Committed: {len(committed_files)}")
            if generated_files - committed_files:
                print(f"Files found only in generated (reference) files: {generated_files - committed_files}")
            if committed_files - generated_files:
                print(f"Files found only in committed files: {committed_files - generated_files}")
            sys.exit(1)

    # Assert that each file has a pair
    if generated_files != committed_files:
        print(f"Error: One or more .pyi files are missing from the PR: {generated_files - committed_files}")
        print(f"Error: DEBUGOne or more extra .pyi files are present in the PR: {committed_files - generated_files}")
        sys.exit(1)

    # Compare file contents
    outdated_files = []
    for relative_path in generated_files:
        if relative_path in SKIPS:
            continue
        generated_file: str = os.path.join(generated_dir, relative_path)
        committed_file: str = os.path.join(committed_dir, relative_path)
        print(f"[Debug] Comparing: {generated_file} with {committed_file}")
        if not filecmp.cmp(generated_file, committed_file, shallow=False):
            with open(generated_file, 'r') as gen_file, open(committed_file, 'r') as com_file:
                gen_lines = [line.lstrip() for line in gen_file.readlines()]
                com_lines = [line.lstrip() for line in com_file.readlines()]
                diff = list(difflib.unified_diff(
                    gen_lines, com_lines,
                    fromfile=f"Generated: {relative_path}",
                    tofile=f"Committed: {relative_path}",
                    lineterm=''
                ))
            if diff:
                changes = [
                    line.lstrip('+- ') for line in diff
                    if (line.startswith('+') or line.startswith('-')) 
                    and not line.startswith(('+++ Committed:', '--- Generated:'))
                    and line.strip()
                ]
                if changes and not all(change.startswith(("import", "from", "__all__")) for change in changes):
                    print(f"Adding diff for {relative_path}. The changes var is: {changes}")
                    outdated_files.append((relative_path, "".join(diff)))

    # Display all outdated files and their diffs
    if outdated_files:
        print("Error: The following stub files are outdated:")
        for relative_path, diff in outdated_files:
            print(f"\nOutdated file: {relative_path}")
            print(diff)
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

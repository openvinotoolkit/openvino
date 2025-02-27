# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import re

def normalize_imports(lines):
    normalized_lines = []
    import_block = []

    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("import ") or stripped_line.startswith("from "):
            import_block.append(stripped_line)
        else:
            if import_block:
                import_block.sort()
                normalized_lines.extend(import_block)
                import_block = []
            normalized_lines.append(stripped_line)

    if import_block:
        import_block.sort()
        normalized_lines.extend(import_block)

    return normalized_lines

def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    normalized_lines1 = normalize_imports(lines1)
    normalized_lines2 = normalize_imports(lines2)

    if normalized_lines1 != normalized_lines2:
        # Ignore differences in memory addresses like in function _get_node_factory at 0x7fdad9f53640
        def remove_memory_addresses(lines):
            return [re.sub(r'at 0x[0-9a-fA-F]+', 'at <memory_address>', line) for line in lines]

        normalized_lines1 = remove_memory_addresses(normalized_lines1)
        normalized_lines2 = remove_memory_addresses(normalized_lines2)

    return normalized_lines1 == normalized_lines2

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: compare_pyi_files.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    if compare_files(file1, file2):
        sys.exit(0)
    else:
        sys.exit(1)

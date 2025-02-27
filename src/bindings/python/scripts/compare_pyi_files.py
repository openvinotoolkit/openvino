# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
import re

# Ignore differences in import order
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


# Ignore differences in memory addresses like in function _get_node_factory at 0x7fdad9f53640
# Also ignore versions like __version__: str = '2025.1.0-18198-298262e2962-pyi_files', which may be different in CI environment
def remove_memory_addresses_and_versions(lines):
    lines = [re.sub(r'at 0x[0-9a-fA-F]+', 'at <memory_address>', line) for line in lines]
    lines = [re.sub(r'__version__: str = \'[\d\.\-a-f]+\'', '__version__: str = \'<version>\'', line) for line in lines]
    return lines


def compare_files(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    # Handle edge cases
    if lines1 != lines2:
        normalized_lines1 = normalize_imports(lines1)
        normalized_lines2 = normalize_imports(lines2)

        normalized_lines1 = remove_memory_addresses_and_versions(normalized_lines1)
        normalized_lines2 = remove_memory_addresses_and_versions(normalized_lines2)
        return normalized_lines1 == normalized_lines2
    else:
        return True


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

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Copyright header checker for OpenVINO project.
Checks that source files have the correct copyright header.
Handles C++ files (.cpp, .hpp, .h) and Python files (.py).
"""

import difflib
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# File type configuration: (extensions, comment_style, has_extra_comment_line)
FILE_TYPES = {
    'cpp': (['.cpp', '.hpp', '.h'], '//', True),
    'python': (['.py'], '#', False),
}


def get_file_type_info(file_path: str) -> Optional[Tuple[str, bool]]:
    """Get (comment_style, has_extra_line) for a file, or None if not supported."""
    ext = Path(file_path).suffix
    for extensions, comment_style, has_extra_line in FILE_TYPES.values():
        if ext in extensions:
            return comment_style, has_extra_line
    return None


def is_supported_file_type(file_path: str) -> bool:
    """Check if this file type should have copyright headers validated."""
    return get_file_type_info(file_path) is not None


def get_encoding_line_number(file_path: str) -> int:
    """
    Detect encoding declaration in Python files.
    Returns the line number (0-indexed) if found, -1 otherwise.
    Encoding must be in the first or second line per PEP 263.
    """
    if not file_path.endswith('.py'):
        return -1
    
    encoding_pattern = r'^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)'
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            for i in range(2):  # Check first two lines per PEP 263
                line = f.readline()
                if not line:
                    break
                if re.match(encoding_pattern, line):
                    return i
    except Exception:
        pass
    
    return -1


def get_expected_header(file_path: str) -> str:
    """Get the expected copyright header for a file."""
    info = get_file_type_info(file_path)
    if not info:
        return ""
    
    comment_style, has_extra_line = info
    current_year = datetime.now().year
    header = f"{comment_style} Copyright (C) 2018-{current_year} Intel Corporation\n"
    header += f"{comment_style} SPDX-License-Identifier: Apache-2.0\n"
    if has_extra_line:
        header += f"{comment_style}\n"
    return header


def should_check_file(file_path: str) -> bool:
    """Determine if a file should be checked for copyright header."""
    return os.path.isfile(file_path) and is_supported_file_type(file_path)


def get_file_header(file_path: str, num_lines: int = 3) -> str:
    """Read the first few lines of a file, handling BOM and encoding declaration if present."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            all_lines = [f.readline() for _ in range(num_lines + 2)]  # Read extra lines to account for encoding
        
        # For Python files, skip the encoding line when checking copyright
        encoding_line = get_encoding_line_number(file_path)
        if encoding_line >= 0:
            # Skip encoding line and return the copyright-relevant lines
            return ''.join(all_lines[encoding_line + 1:encoding_line + 1 + num_lines])
        
        return ''.join(all_lines[:num_lines])
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}")
        return ""


def check_copyright_header(file_path: str) -> bool:
    """Check if file has correct copyright header."""
    expected = get_expected_header(file_path)
    if not expected:
        return True
    
    content = get_file_header(file_path)
    expected_lines = expected.rstrip('\n').split('\n')
    file_lines = content.rstrip('\n').split('\n')
    
    return len(file_lines) >= len(expected_lines) and \
           all(file_lines[i] == expected_lines[i] for i in range(len(expected_lines)))


def generate_diff(file_path: str) -> str:
    """Generate a unified diff for fixing the copyright header."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            original_lines = f.readlines()
    except Exception as e:
        raise IOError(f"Could not read {file_path}: {e}")

    info = get_file_type_info(file_path)
    if not info:
        return ""

    comment_style, has_extra_line = info
    current_year = datetime.now().year

    encoding_line = get_encoding_line_number(file_path)
    header_start = encoding_line + 1 if encoding_line >= 0 else 0

    has_copyright = any('Copyright' in line or 'SPDX-License-Identifier' in line
                        for line in original_lines[header_start:header_start + 5])

    correct_header = [
        f"{comment_style} Copyright (C) 2018-{current_year} Intel Corporation\n",
        f"{comment_style} SPDX-License-Identifier: Apache-2.0\n",
    ]
    if has_extra_line:
        correct_header.append(f"{comment_style}\n")
    correct_header.append("\n")

    corrected_lines = list(original_lines)
    if has_copyright:
        # Walk the actual contiguous comment block so we never overwrite lines beyond it.
        block_end = header_start
        while block_end < len(original_lines) and original_lines[block_end].startswith(comment_style):
            block_end += 1
        # Include the single trailing blank line that separates the header from code
        if block_end < len(original_lines) and original_lines[block_end].strip() == '':
            block_end += 1
        corrected_lines[header_start:block_end] = correct_header
    else:
        # Insert the header at the right position (after encoding line if present)
        corrected_lines[header_start:header_start] = correct_header

    normalized_path = file_path.lstrip('/')
    diff = difflib.unified_diff(
        [l.rstrip('\n') for l in original_lines],
        [l.rstrip('\n') for l in corrected_lines],
        fromfile=f"a/{normalized_path}",
        tofile=f"b/{normalized_path}",
        lineterm='',
    )
    return '\n'.join(diff) + '\n'


def main():
    if len(sys.argv) < 2:
        print("Usage: check_copyright.py <changed_files.txt>")
        sys.exit(1)
    
    changed_files_list = sys.argv[1]
    
    # Read list of changed files
    try:
        with open(changed_files_list, 'r') as f:
            changed_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: File {changed_files_list} not found")
        sys.exit(1)
    
    # Filter files that should be checked
    files_to_check = [f for f in changed_files if should_check_file(f)]
    
    if not files_to_check:
        print("No C++ or Python source files to check.")
        sys.exit(0)
    
    print(f"Checking {len(files_to_check)} files for copyright headers...\n")
    
    # Check each file and collect issues
    issues = []
    diff_content = []
    
    for file_path in files_to_check:
        is_valid = check_copyright_header(file_path)
        
        if is_valid:
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path}")
            issues.append(file_path)
            try:
                diff_content.append(generate_diff(file_path))
            except IOError as e:
                print(f"Error: {e}")
                print(f"File '{file_path}' was identified as having copyright issues but cannot be read for generating fixes.")
                sys.exit(1)
    
    print()
    
    if not issues:
        print("✅ All files have correct copyright headers!")
        sys.exit(0)
    
    # Report issues and generate fix files
    print(f"Found {len(issues)} file(s) with copyright header issues:")
    for file_path in issues:
        print(f"  - {file_path}")
    print()
    
    # Write list of failed files for GitHub Actions
    with open("copyright_issues.txt", 'w') as f:
        f.write('\n'.join(issues))
    
    # Write combined diff file
    diff_file = "copyright_fixes.diff"
    with open(diff_file, 'w') as f:
        f.write('\n'.join(diff_content))
    
    print(f"Generated diff file: {diff_file}")
    print(f"Apply with: patch -p1 < {diff_file}\n")
    
    sys.exit(1)


if __name__ == "__main__":
    main()

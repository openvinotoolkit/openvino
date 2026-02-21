# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Copyright header checker for OpenVINO project.
Checks that source files have the correct copyright header.
Handles C++ files (.cpp, .hpp, .h) and Python files (.py).
"""

import os
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
    """Read the first few lines of a file, handling BOM if present."""
    try:
        with open(file_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            return ''.join(f.readline() for _ in range(num_lines))
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
            lines = f.read().split('\n')
    except Exception as e:
        raise IOError(f"Could not read {file_path}: {e}")
    
    info = get_file_type_info(file_path)
    if not info:
        return ""
    
    comment_style, _ = info
    expected = get_expected_header(file_path)
    expected_lines = expected.rstrip('\n').split('\n')
    
    # Count how many leading lines belong to the existing copyright header.
    # Include blank lines and empty comments (e.g. "//") that may precede or
    # follow the copyright/SPDX keyword lines.  Only treat the block as a
    # header if it actually contains at least one copyright keyword.
    header_keywords = {'copyright', 'spdx-license-identifier'}
    num_old_header = 0
    has_keyword = False
    for line in lines:
        stripped = line.strip()
        is_blank = stripped == ''
        is_comment = stripped.startswith(comment_style)
        if not is_blank and not is_comment:
            break
        if is_comment:
            comment_text = stripped[len(comment_style):].strip().lower()
            if any(kw in comment_text for kw in header_keywords):
                has_keyword = True
            elif comment_text != '':
                # Non-empty comment that isn't a copyright keyword — stop
                break
        num_old_header += 1
    
    # If the leading block didn't contain any copyright keywords, it's
    # just regular blank lines / comments — don't treat it as a header.
    if not has_keyword:
        num_old_header = 0
    
    diff_lines = [f"--- a/{file_path}", f"+++ b/{file_path}"]
    
    if num_old_header > 0:
        # Replace the existing (possibly partial) header with the full expected one
        diff_lines.append(f"@@ -1,{num_old_header} +1,{len(expected_lines)} @@")
        for i in range(num_old_header):
            diff_lines.append(f"-{lines[i]}")
        for exp_line in expected_lines:
            diff_lines.append(f"+{exp_line}")
    else:
        # Missing copyright - insert at beginning
        num_insert = len(expected_lines) + 1  # +1 for blank separator line
        diff_lines.append(f"@@ -0,0 +1,{num_insert} @@")
        for exp_line in expected_lines:
            diff_lines.append(f"+{exp_line}")
        diff_lines.append("+")
    
    return '\n'.join(diff_lines) + '\n'


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

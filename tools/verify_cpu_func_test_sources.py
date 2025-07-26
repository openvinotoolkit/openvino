#!/usr/bin/env python3
"""
Verification script for CPU functional tests source files.
Compares the explicit file lists from CMakeLists.txt files with what would be found by GLOB_RECURSE.
"""

import os
import glob
import sys
import subprocess
import tempfile
from pathlib import Path


def get_glob_recurse_files(base_dir):
    """Get files that would be found by file(GLOB_RECURSE)"""
    extensions = ['*.cpp', '*.c', '*.hpp', '*.h', '*.inl', '*.S']
    files = []
    
    for ext in extensions:
        pattern = os.path.join(base_dir, '**', ext)
        files.extend(glob.glob(pattern, recursive=True))
    
    # Convert to relative paths and sort
    rel_files = []
    for f in files:
        rel_path = os.path.relpath(f, base_dir)
        rel_files.append(rel_path)
    
    return sorted(rel_files)


def get_cmake_explicit_files(base_dir):
    """Get files from the explicit CMakeLists.txt files using cmake -P"""
    
    # Create a temporary cmake script to extract the property
    cmake_script = f"""
cmake_minimum_required(VERSION 3.21)
project(test_extract)

# Set the source directory
set(CMAKE_CURRENT_SOURCE_DIR "{base_dir}")

# Initialize the global property
set_property(GLOBAL PROPERTY CPU_FUNC_TESTS_SRC "")

# Simulate the subdirectory inclusion logic without actually including files
function(add_subdirectory_simulation dir)
    # This is a mock function - in reality we'd process the CMakeLists.txt files
endfunction()

# We need to actually execute the CMakeLists.txt files to get the property
# This is a simplified version - in practice we'd need to process all the CMakeLists.txt files

# For now, let's just extract from the existing structure
get_property(ALL_FILES GLOBAL PROPERTY CPU_FUNC_TESTS_SRC)
foreach(file IN LISTS ALL_FILES)
    file(RELATIVE_PATH rel_file "{base_dir}" "${{file}}")
    message("FILE: ${{rel_file}}")
endforeach()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cmake', delete=False) as f:
        f.write(cmake_script)
        cmake_script_path = f.name
    
    try:
        # Run cmake -P to execute the script
        result = subprocess.run(
            ['cmake', '-P', cmake_script_path],
            capture_output=True,
            text=True,
            cwd=base_dir
        )
        
        # Extract files from output
        explicit_files = []
        for line in result.stdout.split('\n'):
            if line.startswith('FILE: '):
                file_path = line[6:].strip()
                explicit_files.append(file_path)
        
        return sorted(explicit_files)
    
    finally:
        os.unlink(cmake_script_path)


def get_cmake_files_simple(base_dir):
    """Simple extraction of files from CMakeLists.txt files"""
    explicit_files = []
    
    # Find all CMakeLists.txt files
    for root, dirs, files in os.walk(base_dir):
        if 'CMakeLists.txt' in files:
            cmake_path = os.path.join(root, 'CMakeLists.txt')
            
            # Skip the main CMakeLists.txt
            if cmake_path == os.path.join(base_dir, 'CMakeLists.txt'):
                continue
            
            # Parse the CMakeLists.txt to extract file lists
            with open(cmake_path, 'r') as f:
                content = f.read()
                
            # Look for set(<var>_SRC blocks
            lines = content.split('\n')
            in_src_block = False
            current_dir = os.path.relpath(root, base_dir)
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('set(') and '_SRC' in line:
                    in_src_block = True
                    continue
                
                if in_src_block:
                    if line == ')':
                        in_src_block = False
                        continue
                    
                    if line and not line.startswith('#'):
                        # Extract filename
                        filename = line.strip()
                        if filename.endswith('.cpp') or filename.endswith('.hpp') or filename.endswith('.h') or filename.endswith('.c') or filename.endswith('.inl') or filename.endswith('.S'):
                            full_path = os.path.join(current_dir, filename)
                            # Normalize path separators
                            full_path = full_path.replace('\\', '/')
                            explicit_files.append(full_path)
    
    return sorted(explicit_files)


def compare_file_lists(glob_files, explicit_files):
    """Compare the two file lists and report differences"""
    glob_set = set(glob_files)
    explicit_set = set(explicit_files)
    
    missing_in_explicit = glob_set - explicit_set
    extra_in_explicit = explicit_set - glob_set
    
    success = len(missing_in_explicit) == 0 and len(extra_in_explicit) == 0
    
    if success:
        print("✓ All source files match between GLOB_RECURSE and explicit lists")
        print(f"  Total files: {len(glob_files)}")
        return True
    else:
        print("✗ Mismatch found between GLOB_RECURSE and explicit lists")
        print(f"  GLOB_RECURSE found: {len(glob_files)} files")
        print(f"  Explicit lists have: {len(explicit_files)} files")
        
        if missing_in_explicit:
            print(f"\n  Missing in explicit lists ({len(missing_in_explicit)} files):")
            for f in sorted(missing_in_explicit):
                print(f"    - {f}")
        
        if extra_in_explicit:
            print(f"\n  Extra in explicit lists ({len(extra_in_explicit)} files):")
            for f in sorted(extra_in_explicit):
                print(f"    + {f}")
        
        return False


def main():
    """Main verification function"""
    base_dir = "/home/aobolens/prog/openvino3/src/plugins/intel_cpu/tests/functional"
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return 1
    
    print("Verifying CPU functional tests source files...")
    print(f"Base directory: {base_dir}")
    
    # Get files from GLOB_RECURSE simulation
    print("\n1. Getting files that would be found by GLOB_RECURSE...")
    glob_files = get_glob_recurse_files(base_dir)
    print(f"   Found {len(glob_files)} files")
    
    # Get files from explicit CMakeLists.txt files
    print("\n2. Getting files from explicit CMakeLists.txt files...")
    explicit_files = get_cmake_files_simple(base_dir)
    print(f"   Found {len(explicit_files)} files")
    
    # Compare the lists
    print("\n3. Comparing file lists...")
    success = compare_file_lists(glob_files, explicit_files)
    
    if success:
        print("\n✓ Verification passed - all source files are properly listed")
        return 0
    else:
        print("\n✗ Verification failed - file lists don't match")
        return 1


if __name__ == "__main__":
    sys.exit(main())
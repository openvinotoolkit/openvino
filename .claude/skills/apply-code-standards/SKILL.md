---
name: apply-code-standards
description: Detect and fix clang-format, clang-tidy, and copyright header violations in an OpenVINO C++ codebase. Use when the user complains about code style or formatting, asks to clean up changes, fix linting, add a copyright header, or when a style check or linting CI job is failing. Do not use for build errors, compilation failures, linker errors, test failures, runtime crashes, accuracy issues, or CMake config problems.
---

# Apply Code Standards

Iteratively detect and fix all clang-format, clang-tidy, and copyright violations introduced by the current branch's changes.

> Background on each tool (flags, targets, install instructions, recommended fix order): [docs/dev/coding_style.md](../../../docs/dev/coding_style.md).

## Step 1: Identify Affected Build Targets

Ask the user which files to check:
> "Should I check uncommitted changes only, or all changes on the current branch? If the full branch, what is the upstream branch to diff against (e.g. `origin/master`)?"

For **uncommitted changes** (staged + unstaged):
```bash
{ git diff --name-only HEAD; git diff --name-only --cached; } | sort -u | grep -v '^thirdparty' | tee /tmp/changed_files.txt
```

For **full branch** (all commits since upstream):
```bash
git diff --name-only <upstream> | grep -v '^thirdparty' | tee /tmp/changed_files.txt
```

Map them to CMake targets. If the mapping is not obvious, ask the user:
> "Which CMake targets cover the files you changed? (e.g., `openvino_intel_cpu_plugin`)"

## Step 2: Configure with clang-18 + Style Checks

clang-tidy requires the build to use clang-18 as the compiler. Check whether an existing build directory already uses clang-18:

```bash
grep "CMAKE_CXX_COMPILER" <build_dir>/CMakeCache.txt | grep -i clang
```

- **If it does**, re-run cmake in it with the style flags appended — do not wipe the existing configuration:
  ```bash
  cmake -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=ON <build_dir>
  ```
- **If it doesn't** (or no build directory exists), create a dedicated `build-clang/` directory:
  ```bash
  mkdir -p build-clang && cd build-clang
  cmake -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_C_COMPILER=clang-18 \
        -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=ON \
        -DCMAKE_BUILD_TYPE=Release ..
  ```

See [docs/dev/coding_style.md](../../../docs/dev/coding_style.md) for full flag reference.

## Step 3: Check, Fix, Repeat

Follow the fix order from `coding_style.md`: copyright → clang-tidy → clang-format. Repeat until all checks pass.

### 3a. Copyright headers

Run the check script and patch workflow per [coding_style.md](../../../docs/dev/coding_style.md). If the patch looks wrong, fix the header directly in the file.

### 3b. clang-tidy

Build the affected targets (pipe to `/tmp/ct_check.txt`), then separate the two issue categories for your changed files:
```bash
# Compilation errors (no check name in brackets)
grep "error:" /tmp/ct_check.txt | grep -v "\[" | grep -Ff /tmp/changed_files.txt

# clang-tidy diagnostics (have check name in brackets, e.g. [modernize-use-auto])
grep -E "warning:|error:" /tmp/ct_check.txt | grep "\[" | grep -Ff /tmp/changed_files.txt
```

**Fix compilation errors first:** `ENABLE_CLANG_TIDY_FIX` cannot help with these.

If a tidy diagnostic cannot be resolved, **do not add `// NOLINT` on your own** — present it to the user and ask whether to suppress, refactor, or handle it another way.

### 3c. clang-format

Use `clang_format_check_all` to detect violations, `clang_format_fix_all` to auto-fix (see [coding_style.md](../../../docs/dev/coding_style.md)). Pipe output to `/tmp/cf_check.txt` and grep for `"Code style check failed"`. Re-run the check after fixing to confirm clean.

### 3d. Stop condition

When the copyright check passes, `clang_format_check_all` produces no "Code style check failed" lines, and the changed-file grep in 3b produces zero output — report success to the user.

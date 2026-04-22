---
name: apply-code-standards
description: Detect and fix clang-format, clang-tidy, and copyright header violations in an OpenVINO C++ codebase. Trigger on: code style/formatting complaints, "cleanup changes", "fix linting", "copyright header missing/wrong", "style check failing", "linting CI failure". DO NOT trigger for build errors, compilation failures, linker errors, test failures, runtime crashes, accuracy issues, or CMake config problems.
---

# Apply Code Standards

Iteratively detect and fix all clang-format, clang-tidy, and copyright violations introduced by the current branch's changes.

## Step 1: Identify Affected Build Targets

Detect the upstream base branch:

```bash
git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || git rev-parse --abbrev-ref origin/HEAD 2>/dev/null
```

List changed files (excluding submodules) and write to a temp file for reuse across steps:

```bash
git diff --name-only <upstream> | grep -v '^thirdparty' | tee /tmp/changed_files.txt
```

Map them to CMake targets. If the mapping is not obvious, ask the user:
> "Which CMake targets cover the files you changed? (e.g., `openvino_intel_cpu_plugin`)"

## Step 2: Configure with clang-18 + Style Checks

Locate or create a build directory (typically `build/` or `build-clang/` in the repo root).

Run CMake with clang-18 and both style options enabled:

```bash
cmake -S <repo_root> -B <build_dir> \
  -DCMAKE_C_COMPILER=clang-18 \
  -DCMAKE_CXX_COMPILER=clang++-18 \
  -DENABLE_CLANG_FORMAT=ON \
  -DENABLE_CLANG_TIDY=ON \
  [any other flags already in use by the user's build]
```

If the user already has a configured build directory, re-run cmake in it with just the three new flags appended — do not wipe their existing configuration.

## Step 3: Check, Fix, Repeat

Repeat steps 3a–3d until all checks pass.

### 3a. Copyright headers

**Check:**
```bash
python3 .github/scripts/check_copyright.py /tmp/changed_files.txt
```

**Auto-fix:** If non-zero exit, inspect the generated patch before applying — it is usually correct but can occasionally be malformed:
```bash
cat copyright_fixes.diff
patch -p1 < copyright_fixes.diff
python3 .github/scripts/check_copyright.py /tmp/changed_files.txt
```

**Manual fix:** If the patch looks wrong, fix the copyright header directly in the file instead.

### 3b. clang-tidy

**Check:** Build affected targets as a background task. For multiple targets, repeat `--target`:

```bash
cmake --build <build_dir> --target <target1> --target <target2> -- -j$(nproc) 2>&1 | tee /tmp/ct_check.txt
```

After completion, separate the two categories of issues in your changed files:

```bash
# Compilation errors (no clang-tidy check name in brackets)
grep "error:" /tmp/ct_check.txt | grep -v "\[" | grep -Ff /tmp/changed_files.txt

# clang-tidy diagnostics (have check name in brackets, e.g. [modernize-use-auto])
grep -E "warning:|error:" /tmp/ct_check.txt | grep "\[" | grep -Ff /tmp/changed_files.txt
```

**Fix compilation errors first (manual):** If non-tidy compilation errors are present, read each error and fix the source file by hand. Rebuild to confirm they are resolved before moving on — `ENABLE_CLANG_TIDY_FIX` cannot help with these.

**Auto-fix clang-tidy diagnostics:** Once the code compiles cleanly, re-configure with `ENABLE_CLANG_TIDY_FIX=ON` and rebuild as a background task:

```bash
cmake <build_dir> -DENABLE_CLANG_TIDY_FIX=ON
cmake --build <build_dir> --target <target1> --target <target2> -- -j$(nproc) 2>&1 | tee /tmp/ct_fix.txt
```

Re-check diagnostics after the build. Then reset the flag:
```bash
cmake <build_dir> -DENABLE_CLANG_TIDY_FIX=OFF
```

**Manual fix (remaining tidy issues):** For any tidy diagnostics that auto-fix missed or produced incorrect changes, read the message carefully and fix the source file by hand. If a check fires repeatedly and cannot be fixed, **do not add `// NOLINT` suppressions on your own** — present the diagnostic to the user and ask whether to suppress, refactor, or handle it another way.

### 3c. clang-format

**Check:** Run as a background task:
```bash
cmake --build <build_dir> --target clang_format_check_all 2>&1 | tee /tmp/cf_check.txt
```

After completion, check for failures:
```bash
grep "Code style check failed" /tmp/cf_check.txt
```

**Auto-fix:** If failures found, run the fix target as a background task:
```bash
cmake --build <build_dir> --target clang_format_fix_all 2>&1 | tee /tmp/cf_fix.txt
```

Re-run the check to confirm clean.

**Manual fix:** If auto-fix still fails (very rare), fix the specific file manually and re-check.

### 3d. Stop condition

When the copyright check passes, `clang_format_check_all` produces no "Code style check failed" lines, **and** the changed-file grep in 3b produces zero output, stop and report success to the user.

## Notes

- Always prefer `clang-format-18` / `clang-tidy-18` (versioned binaries) over unversioned ones — the project enforces version 18.
- If clang-18 is not installed, inform the user: `sudo apt install clang-18 clang-format-18 clang-tidy-18`.
- Do not modify files outside the set touched by the current branch unless a clang-tidy fix requires it (e.g., a header change). In that case, inform the user.

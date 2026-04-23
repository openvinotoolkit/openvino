---
name: apply-code-standards
description: Detect and fix clang-format, clang-tidy, and copyright header violations in an OpenVINO C++ codebase. Trigger on: code style/formatting complaints, "cleanup changes", "fix linting", "copyright header missing/wrong", "style check failing", "linting CI failure". DO NOT trigger for build errors, compilation failures, linker errors, test failures, runtime crashes, accuracy issues, or CMake config problems.
---

# Apply Code Standards

Iteratively detect and fix all clang-format, clang-tidy, and copyright violations introduced by the current branch's changes.

> Background on each tool (flags, targets, install instructions, recommended fix order): [docs/dev/coding_style.md](../../../docs/dev/coding_style.md).

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

If the user already has a configured build directory, re-run cmake in it with just the new flags appended — do not wipe their existing configuration.

## Step 3: Check, Fix, Repeat

Follow the fix order from `coding_style.md`: copyright → clang-tidy → clang-format. Repeat until all checks pass.

### 3a. Copyright headers

```bash
python3 .github/scripts/check_copyright.py /tmp/changed_files.txt
```

If non-zero exit, inspect then apply the generated patch:
```bash
cat copyright_fixes.diff
patch -p1 < copyright_fixes.diff
python3 .github/scripts/check_copyright.py /tmp/changed_files.txt
```

If the patch looks wrong, fix the copyright header directly in the file.

### 3b. clang-tidy

Build affected targets as a background task:
```bash
cmake --build <build_dir> --target <target1> --target <target2> -- -j$(nproc) 2>&1 | tee /tmp/ct_check.txt
```

After completion, separate the two issue categories for your changed files:
```bash
# Compilation errors (no check name in brackets)
grep "error:" /tmp/ct_check.txt | grep -v "\[" | grep -Ff /tmp/changed_files.txt

# clang-tidy diagnostics (have check name in brackets, e.g. [modernize-use-auto])
grep -E "warning:|error:" /tmp/ct_check.txt | grep "\[" | grep -Ff /tmp/changed_files.txt
```

**Fix compilation errors first (manual):** Read each error and fix by hand. Rebuild to confirm clean before proceeding — `ENABLE_CLANG_TIDY_FIX` cannot help with these.

**Auto-fix clang-tidy diagnostics:** Once the code compiles cleanly:
```bash
cmake <build_dir> -DENABLE_CLANG_TIDY_FIX=ON
cmake --build <build_dir> --target <target1> --target <target2> -- -j$(nproc) 2>&1 | tee /tmp/ct_fix.txt
cmake <build_dir> -DENABLE_CLANG_TIDY_FIX=OFF
```

**Manual fix (remaining tidy issues):** For diagnostics auto-fix missed or got wrong, fix by hand. If a check cannot be resolved, **do not add `// NOLINT` on your own** — present the diagnostic to the user and ask whether to suppress, refactor, or handle it another way.

### 3c. clang-format

```bash
cmake --build <build_dir> --target clang_format_check_all 2>&1 | tee /tmp/cf_check.txt
grep "Code style check failed" /tmp/cf_check.txt
```

If failures found, auto-fix:
```bash
cmake --build <build_dir> --target clang_format_fix_all 2>&1 | tee /tmp/cf_fix.txt
```

Re-run the check to confirm clean. If auto-fix still fails, fix the specific file manually.

### 3d. Stop condition

When the copyright check passes, `clang_format_check_all` produces no "Code style check failed" lines, and the changed-file grep in 3b produces zero output — report success to the user.

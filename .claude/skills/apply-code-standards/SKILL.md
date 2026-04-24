---
name: apply-code-standards
description: Detect and fix clang-format, clang-tidy, and copyright header violations in an OpenVINO C++ codebase. Use when the user complains about code style or formatting, asks to clean up changes, fix linting, add a copyright header, or when a style check or linting CI job is failing. Do not use for build errors, compilation failures, linker errors, test failures, runtime crashes, accuracy issues, or CMake config problems.
---

# Apply Code Standards

Iteratively detect and fix all clang-format, clang-tidy, and copyright violations introduced by the current branch's changes.

## Step 1: Identify Affected Build Targets

Ask the user to confirm the upstream reference:
> "I'll diff against `upstream/master`. Let me know if you use a different upstream."

```bash
git diff --name-only <ref> | grep -v '^thirdparty' | tee /tmp/changed_files.txt
```

Map them to CMake targets. If the mapping is not obvious, ask the user:
> "Which CMake targets cover the files you changed? (e.g., `openvino_intel_cpu_plugin`)"

## Step 2: Check, Fix, Repeat

Follow the fix order from [coding_style.md](../../../docs/dev/coding_style.md). Repeat until all checks pass.

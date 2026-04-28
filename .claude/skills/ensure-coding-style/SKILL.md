---
name: ensure-coding-style
description: Detect and fix clang-format, clang-tidy, and copyright header violations in an OpenVINO C++ codebase. Use when the user complains about code style or formatting, asks to clean up changes, fix linting, add a copyright header, or when a style check or linting CI job is failing. Do not use for build errors, compilation failures, linker errors, test failures, runtime crashes, accuracy issues, or CMake config problems.
---

# Apply Code Standards

Iteratively detect and fix all clang-format, clang-tidy, and copyright violations introduced by the current branch's changes.

## Step 1: Identify Affected Files and Build Targets

**Before running any commands**, check whether the upstream reference is already confirmed in the conversation. If it is not, stop and ask:

> "I'll diff against `upstream/master`. Let me know if you use a different upstream."

Do not proceed until the user replies. Use the confirmed reference <ref_branch> in all subsequent steps. Once the upstream branch is known, you must fetch <ref_branch> and find a merge-base commit <ref_commit>, against which the diff should be collected. Then run:

```bash
git diff --name-only <ref_commit> | grep -v '^thirdparty' | tee /tmp/changed_files.txt
```

Map the changed files to CMake targets. If the mapping is not obvious, ask the user:

> "Which CMake targets cover the files you changed? (e.g., `openvino_intel_cpu_plugin`)"

## Step 2: Check, Fix, Repeat

**Follow the fix order and tool instructions from [coding_style.md](../../../docs/dev/coding_style.md) exactly  — do not rearrange or skip steps.** Read that file before proceeding. Repeat the full cycle until all checks pass.

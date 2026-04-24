# OpenVINO Coding Guidelines

## Fix order

When bringing a set of changes into compliance, address the checks in the following order:

1. **Copyright headers** — fast, no build required
2. **clang-tidy** — requires a clean build; fix compilation errors first, then apply tidy auto-fixes
3. **Naming style (ncc)** — requires clang, but no full rebuild; fix after tidy to avoid redundant churn
4. **clang-format** — purely cosmetic; run last so tidy auto-fixes don't reintroduce formatting issues

## Copyright headers

All source files must carry the standard OpenVINO copyright header.

C++ (`.cpp`, `.hpp`, `.h`):
```cpp
// Copyright (C) 2018-<current_year> Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
```

Python (`.py`):
```python
# Copyright (C) 2018-<current_year> Intel Corporation
# SPDX-License-Identifier: Apache-2.0
```

To check and fix headers automatically, the script at `.github/scripts/check_copyright.py` can be used. If it exits non-zero it writes `copyright_fixes.diff`, which can be applied with `patch -p1 < copyright_fixes.diff`.

Note: automatically generated patch can occasionally produce incorrect changes — in some cases, the manual fix is required.

## Static analysis (clang-tidy)

`clang-tidy-18` is used for static analysis and enforcing modern C++ patterns.

**Prerequisites:** `clang-tidy-18` must be installed, and the project must be compiled with clang as the compiler. Before enabling it, verify that an existing build directory already uses clang:
```bash
grep "CMAKE_CXX_COMPILER" <build_dir>/CMakeCache.txt | grep -i clang
```

**If it does**, re-run cmake in it to append the style flags (preserves the existing configuration):
```bash
cmake -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=ON <build_dir>
```

**If it doesn't** (or no build directory exists), create a dedicated `build-clang/` directory:
```bash
mkdir -p build-clang && cd build-clang
cmake -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_C_COMPILER=clang-18 \
      -DENABLE_CLANG_FORMAT=ON -DENABLE_CLANG_TIDY=ON \
      -DCMAKE_BUILD_TYPE=Release ..
```

Build a target to run checks (warnings appear inline during compilation):
```bash
cmake --build . --target <your_target> -- -j$(nproc)
```

To auto-fix diagnostics, enable the fix mode and rebuild:
```bash
cmake -DENABLE_CLANG_TIDY_FIX=ON .
cmake --build . --target <your_target> -- -j$(nproc)
```

Notes:
- Compilation errors must be resolved before clang-tidy diagnostics can be applied.
- Auto-fix can occasionally produce incorrect changes — in some cases, the manual fix is required.
- It is better to use a dedicated build directory for clang-tidy (e.g. `build-clang/`) — clang-tidy rewrites compilation databases and can interfere with incremental builds in a shared directory.
- Do not use `// NOLINT` suppressions to silence diagnostics. A suppression hides the issue without fixing it and can mask real problems in future code. It is strongly recommended to  suppress if if the check is a confirmed false positive with no correct alternative, and document the reason inline.

## Naming style (ncc)

OpenVINO has strict rules for naming style in public API. All classes must start with a capital letter, and methods and functions should be named in `snake_case` style.
To check the naming style, `ncc` tool is integrated in the OpenVINO. Read the detailed information about the naming style can be found in the [configuration file](../../cmake/developer_package/ncc_naming_style/openvino.style).
To activate this tool, you need to have `clang` on the local machine and enable the CMake option `ENABLE_NCC_STYLE`.
After that, you can use the `ncc_all` target to check the naming style.

## Coding style (clang-format)

The majority of OpenVINO components use `clang-format-18` for code style checks.

**Prerequisites:** `clang-format-18` must be installed. No specific compiler is required — clang-format runs as a standalone tool independent of how the project was compiled.

The code style is based on the Google Code style with some differences. All the differences are described in the configuration file:
https://github.com/openvinotoolkit/openvino/blob/69f709028a5f8da596d1d0df9a0101e517c35708/src/.clang-format#L1-L28

To enable clang-format checks in the build, set the corresponding CMake flag:
```bash
cmake -DENABLE_CLANG_FORMAT=ON <other flags> ..
```

Check and fix targets:
```bash
cmake --build . --target clang_format_check_all   # report violations
cmake --build . --target clang_format_fix_all     # auto-fix violations
```

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)

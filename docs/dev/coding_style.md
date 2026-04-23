# OpenVINO Coding Guidelines

## Coding style (clang-format)

The majority of OpenVINO components use `clang-format-18` for code style checks.

The code style is based on the Google Code style with some differences. All the differences are described in the configuration file:
https://github.com/openvinotoolkit/openvino/blob/69f709028a5f8da596d1d0df9a0101e517c35708/src/.clang-format#L1-L28

To use clang-format locally, `clang-format-18` should be installed, and the corresponding CMake should be enabled:
```bash
cmake -DENABLE_CLANG_FORMAT=ON <other flags> ..
```

Check and fix targets:
```bash
cmake --build . --target clang_format_check_all   # report violations
cmake --build . --target clang_format_fix_all     # auto-fix violations
```

## Naming style (ncc)

OpenVINO has strict rules for naming style in public API. All classes must start with a capital letter, and methods and functions should be named in `snake_case` style.
To check the naming style, `ncc` tool is integrated in the OpenVINO. Read the detailed information about the naming style can be found in the [configuration file](../../cmake/developer_package/ncc_naming_style/openvino.style).
To activate this tool, you need to have `clang` on the local machine and enable the CMake option `ENABLE_NCC_STYLE`.
After that, you can use the `ncc_all` target to check the naming style.

## Static analysis (clang-tidy)

`clang-tidy-18` is used for static analysis and enforcing modern C++ patterns. To use it locally, `clang-tidy-18` should be installed, and the corresponding CMake should be enabled:
```bash
cmake -DENABLE_CLANG_TIDY=ON <other flags> ..
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
- Use a dedicated build directory for clang-tidy (e.g. `build-clang/`) — clang-tidy rewrites compilation databases and can interfere with incremental builds in a shared directory.

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

## Recommended fix order

When bringing a set of changes into compliance, it is recommended to address the three checks in the following order:

1. **Copyright headers** — fast, no build required
2. **clang-tidy** — requires a clean build; fix compilation errors first, then apply tidy auto-fixes
3. **clang-format** — purely cosmetic; run last so tidy auto-fixes don't reintroduce formatting issues

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)

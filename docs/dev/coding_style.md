# OpenVINO Coding Guidelines

## Coding style

The majority of OpenVINO components use `clang-format-18` for code style check.

The code style is based on the Google Code style with some differences. All the differences are described in the configuration file:
https://github.com/openvinotoolkit/openvino/blob/69f709028a5f8da596d1d0df9a0101e517c35708/src/.clang-format#L1-L28

To fix the code style on your local machine, you need to install the `clang-format-18` tool and make sure that the CMake option `ENABLE_CLANG_FORMAT` is enabled.
If all dependencies are resolved, you can use the `clang_format_fix_all` target to fix all code style issues.

## Naming style

OpenVINO has strict rules for naming style in public API. All classes must start with a capital letter, and methods and functions should be named in `snake_case` style.
To check the naming style, `ncc` tool is integrated in the OpenVINO. Read the detailed information about the naming style can be found in the [configuration file](../../cmake/developer_package/ncc_naming_style/openvino.style).
To activate this tool, you need to have `clang` on the local machine and enable the CMake option `ENABLE_NCC_STYLE`.
After that, you can use the `ncc_all` target to check the naming style.

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)

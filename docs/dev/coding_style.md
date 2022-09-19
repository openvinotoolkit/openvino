# OpenVINO Coding Guidelines

## Coding style

Majority of OpenVINO components use `clang-format-9` for code style check.

The code style is based on Google Code style with some differences. All differences are described in the configuration file:
https://github.com/openvinotoolkit/openvino/blob/69f709028a5f8da596d1d0df9a0101e517c35708/src/.clang-format#L1-L28

To fix code style on your local machine, you need to have installed `clang-format-9` tool and be sure that CMake option `ENABLE_CLANG_FORMAT` is enabled.
If all dependencies are resolved, `clang_format_fix_all` target can be used to fix all code style issues.

## Naming style

OpenVINO has a strict rules for naming style in public API. All classes should be started from capital letter, methods and functions are named in `snake_case` style.
To check the naming style `ncc` tool is integrated inside the OpenVINO. The detailed information about naming style can be found in the [configuration file](../../cmake/developer_package/ncc_naming_style/openvino.style).
In order to activate this tool you need to have `clang` on the local machine and enabled CMake option `ENABLE_NCC_STYLE`.
After that `ncc_all` target can be used to check the naming style.

## See also
 * [OpenVINO™ README](../../README.md)
 * [Developer documentation](../../docs/dev/index.md)

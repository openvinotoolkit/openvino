# OpenVINO Conditional Compilation


OpenVINO Conditional Compilation(CC) is a very useful feature for the scenarios that is senstive for binaries size, it can significantly optimize OpenVINO™ binaries size by excluding unnecessary code regions by the help of ITT profiler, especially when build application with static OpenVINO package.

## Key contacts

People from [openvino-ie-maintainers]https://github.com/orgs/openvinotoolkit/teams/openvino-ie-maintainers group have the rights to approve and merge PRs related with conditional compilation.

## Components

* [docs](./docs/) contains documentation for conditional compilation.
* [include](./include/) contains header files that define class and macros for conditional compilation.
* [scripts](./scripts/) script tools that used in condition compilation.

## Tutorials

* [Conditional Compilation Introduction](../../../docs/dev/conditional_compilation.md)
* [Develop CC for New Components](./docs/develop_cc_for_new_component.md)

## How to contribute

See [CONTRIBUTING](../../../CONTRIBUTING.md) for details.

## See also

* [OpenVINO™ README](../../../README.md)
* [Developer documentation](../../../docs/dev/index.md)
* [OpenVINO Wiki](https://github.com/openvinotoolkit/openvino/wiki#how-to-build)

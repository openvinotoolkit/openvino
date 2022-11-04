# OpenVINO conditional compilation

OpenVINO conditional compilation(CC) can greatly optimize package binaries size for particular one or multiple models by excluding unnecessary components or code region. There are 2 CC modes for different CC build stages: SELECTIVE_BUILD_ANALYZER and SELECTIVE_BUILD.

`SELECTIVE_BUILD_ANALYZER` enables analyzed mode for annotated code regions, when this process completes, a new C++ header file will be created which contains all macros definition for enabling active code regions, and this header file will be referenced by SELECTIVE_BUILD mode. This mode will be enabled when build OpenVINO with options `-DSELECTIVE_BUILD=COLLECT -DENABLE_PROFILING_ITT=ON` 

`SELECTIVE_BUILD` excludes all annotated inactive code region to be compiled, which has been profiled/analyzed in SELECTIVE_BUILD_ANALYZER mode, so as to generate final optimized binaries without inactive code. It can be enabled when build OpenVINO with option `-DSELECTIVE_BUILD=ON -DENABLE_PROFILING_ITT=OFF  -DSELECTIVE_BUILD_STAT=<cc_data.csv>`

Note: If above options are not enabled, conditional compilation will be OFF and default behavior is kept, all features of OpenVINO are enabled. You can igore `SELECTIVE_BUILD` or set option `-DSELECTIVE_BUILD=OFF` 


## Tutorials

* [OpenVINO Conditional Compilation](../../../docs/dev/conditional_compilation.md)
* [Develop_CC_for_new_component](./docs/develop_cc_for_new_component.md)

## How to contribute to the OpenVINO repository

See [CONTRIBUTING](../../../CONTRIBUTING.md) for details.
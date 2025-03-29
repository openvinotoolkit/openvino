# Performance Analysis Using ITT Counters

## Contents

- [Introduction](#introduction)
- [Performance analysis](#performance-analysis)
- [Adding new ITT counters](#adding-new-itt-counters)

## Introduction

OpenVINO has a powerful capabilities for performance analysis of the key stages, such as read time and load time. Most of the modules and features have been tagged with [Intel ITT](https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis.html) counters, which allows us to measure the performance of these components.

## Performance analysis

For performance analysis, follow the steps below:
1. Run the CMake tool with the following option: `-DENABLE_PROFILING_ITT=ON` and build OpenVINO.
2. Choose the tool for statistics collection using ITT counters.

    1. [Intel Vtune Profiler](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html)

3. Run an OpenVINO project with the performance analysis tool.

### Intel Vtune Profiler

#### Example of running the tool:

```sh
vtune -collect hotspots -k sampling-mode=hw -k enable-stack-collection=true -k stack-size=0 -k sampling-interval=0.5 -- ./benchmark_app -nthreads=1 -api sync -niter 1 -nireq 1 -m ./resnet-50-pytorch/resnet-50-pytorch.xml
```

#### Mandatory parameters:
* -collect hotspots

#### Generated artifacts:
`r000hs`
Generated file can be opened with Vtune client.

## Adding new ITT counters

Use API defined in [openvino/itt](https://docs.openvinotoolkit.org/latest/itt_2include_2openvino_2itt_8hpp.html) module.

## See also

 * [OpenVINO™ README](../../README.md)
 * [OpenVINO Core Components](../README.md)
 * [OpenVINO Plugins](../plugins/README.md)
 * [OpenVINO GPU Plugin](../plugins/intel_gpu/README.md)
 * [OpenVINO CPU Plugin](../plugins/intel_cpu/README.md)
 * [OpenVINO NPU Plugin](../plugins/intel_npu/README.md)
 * [Developer documentation](../../docs/dev/index.md)

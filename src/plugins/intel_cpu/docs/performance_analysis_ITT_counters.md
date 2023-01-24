# Performance analysis using ITT counters

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
    1. [Intel SEAPI](https://github.com/vladislav-volkov/IntelSEAPI) should be built from sources. See the [Readme](https://github.com/vladislav-volkov/IntelSEAPI/blob/master/README.txt) file for details.
    2. [Intel Vtune Profiler](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/vtune-profiler.html)
3. Run OpenVINO project with performance analysis tool.

### Intel SEAPI

#### Example of tool run:
`python ~/tools/IntelSEAPI/runtool/sea_runtool.py -o trace -f gt ! ./benchmark_app -niter 1 -nireq 1 -nstreams 1 -api sync -m ./resnet-50-pytorch/resnest-50-pytorch.xml`

#### Mandatory parameters:
* -o trace – output file name
* -f gt - statistics type to be generated (Google traces)

#### Generated artifacts:
`trace.pid-21725-0.json`
Generated file can be opened with google chrome using "chrome://tracing" URL.

### Intel Vtune Profiler

#### Example of tool run:
`vtune -collect hotspots -k sampling-mode=hw -k enable-stack-collection=true -k stack-size=0 -k sampling-interval=0.5 -- ./benchmark_app -nthreads=1 -api sync -niter 1 -nireq 1 -m ./resnet-50-pytorch/resnet-50-pytorch.xml`

#### Mandatory parameters:
* -collect hotspots

#### Generated artifacts:
`r000hs`
Generated file can be opened with Vtune client.

## Adding new ITT counters

Use API defined in [openvino/itt](https://docs.openvinotoolkit.org/latest/itt_2include_2openvino_2itt_8hpp.html) module.

# Introduction

This document describes the use of '--use_device_mem' option in benchmark_app. It makes performance difference for the platforms where memory access for host memory and device memory are not identical. Discrete GPUs and recent version of iGPU get performance boost from this feature.

# Motivation
You can achieve best GPU performance when input data is placed on device memory. Intel OpenCL supports to specify such placement with USM(Unified Shared Memory) feature. It is recommended to place the input data on device memory if possible. For example, if the input data is decoded from a video stream by GPU, it is recommended to use that directly on GPU. On the other hand, if input data is generated from CPU, it is not possible to use this feature.
The bottom line is that the usage of this feature depends on the application data flow. If possible, please place the input data on device memory.

# Benchmark_app support for device memory
OpenVINO benchmark_app sample contains feature to mimic the behavior of placing input data on device memory. It allocates input and output of the neural network on device memory. You can use feature with use_device_mem option from benchmark_app.

### Restriction of use_device_mem
Currently, benchmark_app does not support to fill input data when use_device_mem is on. Input data is filled with random numbers. It is fine to measure performance for the networks where performance does not depend on the input data. However, if the target network performance depends on the input data, this option might report an incorrect result. For example, some object detection networks contain NMS layer and its execution time depends on the input data. In such detection network, it is not recommended to measure performance with use_device_mem option.

### How to build sample for use_device_mem (on Windows)
The option depends on Intel OpenCL feature of USM memory. To use the option, you need to build sample with OpenCL enabled. Here's steps to build sample application with OpenCL.
1. Setup env variable for compiler and OpenVINO release package
1. \> git clone https://github.com/microsoft/vcpkg
1. \> cd vcpkg
1. \> .\bootstrap-vcpkg.bat
1. \> vcpkg search opencl
1. \> vcpkg install opencl
1. openvino_install\samples\cpp> cmake -DCMAKE_BUILD_TYPE=Release -B build -DCMAKE_TOOLCHAIN_FILE=path/to/vcpkg/scripts/buildsystems/vcpkg.cmake
1. openvino_install\samples\cpp> cmake --build build --config Release --parallel

### How to build sample for use_device_mem (on Ubuntu)
1. \# apt install opencl-c-headers opencl-clhpp-headers
1. Build OpenVINO cpp sample with build script

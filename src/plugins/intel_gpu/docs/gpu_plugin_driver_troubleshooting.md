# Driver Issues Troubleshooting

If you see errors like `[CLDNN ERROR]. clGetPlatformIDs error -1001` when running OpenVINO samples / demos, then most likely you have some issues with OpenCL runtime on your machine. This document contains several hints on what to check and how to troubleshoot such issues.

To make sure that OpenCL runtime is functional on your machine, you can use [clinfo](https://github.com/Oblomov/clinfo) tool. On many linux distributions it can be installed via package manager. If it is not available for your system, it can be easily built from sources.

Example of clinfo output:
```
Number of platforms                               1
  Platform Name                                   Intel(R) OpenCL HD Graphics
  Platform Vendor                                 Intel(R) Corporation

  ...

  Platform Name                                   Intel(R) OpenCL HD Graphics
Number of devices                                 1
  Device Name                                     Intel(R) Graphics [0x3e92]
  Device Vendor                                   Intel(R) Corporation
  Device Vendor ID                                0x8086
  Device Version                                  OpenCL 3.0 NEO
  Driver Version                                  20.49.0
  Device OpenCL C Version                         OpenCL C 3.0
  Device Type                                     GPU
```
## 1. Make sure that you have GPU on your system

Some Intel® CPUs might not have integrated GPU, so if you want to run OpenVINO on iGPU, go to [ark.intel website](https://ark.intel.com/) and make sure that your CPU has it.

## 2. Make sure that OpenCL® Runtime is installed

OpenCL runtime is a part of the GPU driver on Windows, but on Linux it should be installed separately. For the installation tips, refer to [OpenVINO docs](https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-linux.html) and [OpenCL Compute Runtime docs](https://github.com/intel/compute-runtime/tree/master/opencl/doc).
To get the support of Intel® Iris® Xe MAX Graphics with Linux, follow the [driver installation guide](https://dgpu-docs.intel.com/devices/iris-xe-max-graphics/index.html)

## 3. Make sure that user has all required permissions to work with GPU device

Add the current Linux user to the `video` group:
```
sudo usermod -a -G video "$(whoami)"
```

## 4. Make sure that iGPU is enabled

```
$ cat /sys/devices/pci0000\:00/0000\:00\:02.0/enable
1
```

## 5. Make sure that "/etc/OpenCL/vendors/intel.icd" contains proper paths to the OpenCL driver

```
$ cat /etc/OpenCL/vendors/intel.icd
/usr/lib/x86_64-linux-gnu/intel-opencl/libigdrcl.so
```
Note: path to the runtime lib may vary in different driver versions

## 6. Use LD_DEBUG=libs to trace loaded libraries

For more details, see the [OpenCL on Linux](https://github.com/bashbaug/OpenCLPapers/blob/markdown/OpenCLOnLinux.md)

## 7. If you are using dGPU with XMX, ensure that HW_MATMUL feature is recognized

OpenVINO contains *hello_query_device* sample application: [link](https://docs.openvino.ai/2024/learn-openvino/openvino-samples/hello-query-device.html)

With this option, you can check whether Intel XMX(Xe Matrix Extension) feature is properly recognized or not. This is a hardware feature to accelerate matrix operations and available on some discrete GPUs.

```
$ ./hello_query_device.py
...
[ INFO ]                OPTIMIZATION_CAPABILITIES: FP32, BIN, FP16, INT8, GPU_HW_MATMUL
```

## 8. If you have errors with OpenCL headers in application build
OpenCL headers should be installed in your system to build application using OpenCL objects. OpenVINO source code distribution contains OpenCL headers thirdparty/ocl/cl_headers. Alternatively you can
install them from [OpenCL Git](https://github.com/KhronosGroup/OpenCL-Headers). To ensure compatibility, make sure that the installed version of OpenCL headers had been released before the OpenVINO version you are using.

## See also
 * [Overview for OpenCL on Linux and troubleshoot](https://bashbaug.github.io/opencl/2019/07/06/OpenCL-On-Linux.html)
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)

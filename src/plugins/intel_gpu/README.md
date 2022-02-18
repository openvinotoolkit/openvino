
### Attached licenses
GPU plugin uses 3<sup>rd</sup>-party components licensed under following licenses:
- *googletest* under [Google\* License](https://github.com/google/googletest/blob/master/googletest/LICENSE)
- *OpenCL™ ICD and C++ Wrapper* under [Khronos™ License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt)
- *RapidJSON* under [Tencent\* License](https://github.com/Tencent/rapidjson/blob/master/license.txt)

## Support
Please report issues and suggestions
[GitHub issues](https://github.com/openvinotoolkit/openvino/issues).

## How to Contribute
We welcome community contributions to GPU plugin. If you have an idea how to improve the library:

- Share your proposal via
 [GitHub issues](https://github.com/openvinotoolkit/openvino/issues)
- Ensure you can build the product and run all the tests with your patch
- In the case of a larger feature, create a test
- Submit a [pull request](https://github.com/openvinotoolkit/openvino/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged into our GitHub repository.

## System Requirements
GPU plugin supports Intel® HD Graphics and Intel® Iris® Graphics and is optimized for Gen9-Gen12LP architectures

GPU plugin currently uses OpenCL™ with multiple Intel OpenCL™ extensions and requires Intel® Graphics Driver to run.

GPU plugin requires CPU with Intel® SSE/Intel® AVX support.

---

The software dependencies are:
- [CMake\*](https://cmake.org/download/) 3.5 or later
- C++ compiler with C++11 standard support compatible with:
    * GNU\* Compiler Collection 4.8 or later
    * clang 3.5 or later
    * [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 17.0 or later
    * Visual C++ 2015 (MSVC++ 19.0) or later

> Intel® CPU intrinsics header (`<immintrin.h>`) must be available during compilation.

- [python™](https://www.python.org/downloads/) 2.7 or later (scripts are both compatible with python™ 2.7.x and python™ 3.x)

# Trademark Information

Intel, the Intel logo, Intel Atom, Intel Core, Intel Xeon Phi, Iris, OpenVINO,
the OpenVINO logo, Pentium, VTune, and Xeon are trademarks
of Intel Corporation or its subsidiaries.

\* Other names and brands may be claimed as the property of others.

Microsoft, Windows, and the Windows logo are trademarks, or registered
trademarks of Microsoft Corporation in the United States and/or other
countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.

Copyright © 2021, Intel Corporation


# Compute Library for Deep Neural Networks (clDNN)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
![v1.0](https://img.shields.io/badge/1.0-RC1-green.svg)

*Compute Library for Deep Neural Networks* (*clDNN*) is an open source performance
library for Deep Learning (DL) applications intended for acceleration of
DL Inference on Intel® Processor Graphics – including HD Graphics and
Iris® Graphics.
*clDNN* includes highly optimized building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. We created
this project to enable the DL community to innovate on Intel® processors.

**Usages supported:** Image recognition, image detection, and image segmentation.

**Validated Topologies:** AlexNet\*, VGG(16,19)\*, GoogleNet(v1,v2,v3)\*, ResNet(50,101,152)\* Faster R-CNN\*, Squeezenet\*, SSD_googlenet\*, SSD_VGG\*, PVANET\*, PVANET_REID\*, age_gender\*, FCN\* and yolo\*.

As with any technical preview, APIs may change in future updates.

## License
clDNN is licensed is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

### Attached licenses
clDNN uses 3<sup>rd</sup>-party components licensed under following licenses:
- *googletest* under [Google\* License](https://github.com/google/googletest/blob/master/googletest/LICENSE)
- *OpenCL™ ICD and C++ Wrapper* under [Khronos™ License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt)
- *RapidJSON* under [Tencent\* License](https://github.com/Tencent/rapidjson/blob/master/license.txt)

## Documentation
The latest clDNN documentation is at [GitHub pages](https://intel.github.io/clDNN/index.html).

There is also inline documentation available that can be [generated with Doxygen](#generating-documentation).

Accelerate Deep Learning Inference with Intel® Processor Graphics whitepaper [link](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics).

## Intel® OpenVino™ Toolkit and clDNN

clDNN is released also together with Intel® OpenVino™ Toolkit, which contains:
- *Model Optimizer* a Python*-based command line tool, which imports trained models from popular deep learning frameworks such as Caffe*, TensorFlow*, and Apache MXNet*.
- *Inference Engine* an execution engine which uses a common API to deliver inference solutions on the platform of your choice (for example GPU with clDNN library)

You can find more information [here](https://software.intel.com/en-us/openvino-toolkit/deep-learning-cv).

## OpenVINO specific changes
    New features:
    - added `not` activation type
    - added `depth_to_space` layer
    - new clip options in `detection_output` (cpu impl) and `proposal` layers
    - added eltwise `xor` and `squared_diff` operations
    - added `gather` layer
    - added `bilinear` mode for position sensitive `roi_pooling` layer
    - added `shuffle_channels` layer
    - added `strided_slice` layer
    - added IE gates ordering for lstm layer
    - added `reverse_sequence` layer
    Bug fixes:
    - fixed unknown bool type error in C API
    - fixed non-relu activation fusing with conv_eltwise node
    - fixed infinite performance regression on several topologies
    - minor internal fixes
    - unified the permute order with cldnn's tensor order
    Other:
    - removed boost
    - supported compilation with c++11 only


## Changelog

### Drop 13.1
    New features:
    - added max mode for contract primitive
    - added one_hot primitive
    - optional explicit output data type support for all primitives
    Bug fixes:
    - fix for graph optimizer (crop primitive)
    - fix for processing order (deconvolution primitive)
    - fix for convolution-eltwise primitive
    UX:
    - cache.json is searched in to library directory
    Performance:
    - optimizations for lstm_gemm primitive

### Drop 13.0
    New features:
    - events pool
    - group support in convolution and deconvolution primitives
    - broadcastable inputs support for eltwise primitive
    - asymmetric padding for convolution primitive
    - fused convolution-eltwise primitive (API extension)
    - auto-calculated output shape support for reshape primitive
    - crop support for i8/s8/i32/i64 types
    - broadcast axis support for broadcast primitive
    - logic and comparison operations support for eltwise primitive
    Bug fixes:
    - added required alignment checks for some fc implementations
    - added lstm support for f16 (half) type
    - reorders for fc moved to graph compiler
    - primitive fusing and reorder fixes
    UX:
    - added internal core tests project
    - refactored optimizations pass manager and passes
    Performance:
    - optimized concatenation during upsampling (unpool)
    - IMAD-based optimizations for convolution, fc, eltwise and pooling primitives (i8/s8)
    - convolution-eltwise fusing optimizations
    - partial writes optimizations for block-based kernels

### Drop 12.1
	- gtests code refactor
	- buildbreak fix

### Drop 12.0
    New features:
    - pyramidRoiAlign primitive
    - multiple axes support for reverse mode in index_select
    - eltwise min/max/mod support for i8/i32/i64
    - broadcast support for i32/i64
    Bug fixes:
    - memory leak fixes
    - in-place reshape
    - no padding for output primitives
    UX:
    - RapidJSON library for auto-tune cache
    - less dependencies in program.cpp
    - do not throw error, when device not validated
    - global pooling in c API
    - optimized padding for convolution

### Drop 11.0
    New features:
    - throttle hints
    - extended border and tile
    - GPU implementation of Detection Output
	- More cases for BatchNorm primitive
    Bug fixes:
    - GEMM fix (align with ONNX)
	- memory leak fix in memory pool
	- increase FC precision for fp16 (fp32 accu)
    Performance:
    - cache for new topologies and devices
    - conv1x1 with stride >1 into eltwise optimization

### Drop 10.0
    New features:
    - condition primitive
    - fused convolution with bn and scale (backprop)
    - scale/shit and mean/var as an output in batch norm
    - add LSTM output selection
    Bug fixes:
    - memory pool fixes
    UX:
    - downgrade to cxx11
    - add support for u8 data type in custom primitive
    - library size optimizations
    Performance:
    - in place concatenation optimization
    - conv1x1 with stride >1 into eltwise optimization

### Drop 9.2
	New features
	- local convolution
	- eltwise with strie

### Drop 9.1
    New features:
    - select index primitive
	- gemm primitive
    Bug fixes:
    - fix for output format in fully connected primitive

### Drop 9.0
    New features:
    - log2 activation function
    - support for i32 and i64 types
    - select primitive
	- border primitive
	- tile primitive
    Bug fixes:
    - dilation > input size fix

### Drop 8.0
    New features:
    - lstm primitive
    - average unpooling primitive
    - serialization - dump weights, biases and kernels
    - scale grad for input and weights primitive
    Bug fixes:
    - wrong gws in concatenation
    - int8 layers
    - convolution depthwise bias concatenation
    - params in engine_info
    - mutable_data filler
    - momentum calculation
    UX:
    - kernel selector renaming
    - bfyx_yxfb batched reorder
    - code cleanups
    - primitives allocation order

### Drop 7.0
    New features:
    - support for img_info=4 in proposal_gpu
    - support images format in winograd
    - support for 2 or more inputs in eltwise
    - priority and throttle hints
    - deconvolution_grad_input primitive
    - fc_grad_input and fc_grad_weights primitives
    Bug fixes:
    - tensor fixes (i.e. less operator fix)
    - cascade concat fixes
    - winograd fixes for bfyx format
    - auto-tuning fixes for weights calculation
    UX:
    - memory pool (reusing memory buffers)
    - added choosen kernel name in graph dump
    - flush memory functionality
    Performance:
    - graph optimizations
    - depth-concatenation with fused relu optimization
    - winograd optimizations
    - deconvolution optimizations (i.e bfyx opt)

### Drop 6.0
	New features:
	- fused winograd
	- image support for weights
	- yolo_region primitive support
	- yolo_reorg primitive support
	Bug fixes:
	- winograd bias fix
	- mean subtract fix
	UX:
	- extend graph dumps
	Performance:
	- update offline caches for newer drivers
	- conv1x1 byxf optimization
	- conv1x1 with images
	- cascade depth concatenation fuse optimization

### Drop 5.0
	New features:
	- split primitive
	- upsampling primitive
	- add preliminary Coffe Lake support
	- uint8 weights support
	- versioning
	- offline autotuner cache
	- Winograd phase 1 - not used yet
	Bug fixes:
	- in-place crop optimization bug fix
	- output spatial padding in yxfb kernels fix
	- local work sizes fix in softmax
	- underflow fix in batch normalization
	- average pooling corner case fix
	UX:
	- graph logger, dumps graphwiz format files
	- extended documentation with API diagram and graph compilation steps
	Performance:
	- softmax optimization
	- lrn within channel optimization
	- priorbox optimization
	- constant propagation

### Drop 4.0
	New features:
	- OOOQ execution model implementation
	- depthwise separable convolution implementation
	- kernel auto-tuner implementation
	Bug fixes:
	- dump hidden layer fix
	- run single layer fix
	- reshape fix
	UX:
	- enable RTTI
	- better error handling/reporting
	Performance:
	- lrn optimization
	- dynamic pruning for sparse fc layers
	- reorder optimization
	- concatenation optimization
	- eltwise optimization
	- activation fusing

### Drop 3.0
	Added:
	- kernel selector
	- custom layer
	Changed:
	- performance improvments
	- bug fixes (deconvolution, softmax, reshape)
	- apply fixes from community reported issues

### Drop 2.0
	Added:
	- step by step tutorial
	Changed:
	- perfomance optimization for: softmax, fully connected, eltwise, reshape
	- bug fixes (conformance)

### Drop 1.0
	- initial drop of clDNN

## Support
Please report issues and suggestions
[GitHub issues](https://github.com/01org/cldnn/issues).

## How to Contribute
We welcome community contributions to clDNN. If you have an idea how to improve the library:

- Share your proposal via
 [GitHub issues](https://github.com/01org/cldnn/issues)
- Ensure you can build the product and run all the examples with your patch
- In the case of a larger feature, create a test
- Submit a [pull request](https://github.com/01org/cldnn/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged into our internal and GitHub repositories.

## System Requirements
clDNN supports Intel® HD Graphics and Intel® Iris® Graphics and is optimized for
- Codename *Skylake*:
    * Intel® HD Graphics 510 (GT1, *client* market)
    * Intel® HD Graphics 515 (GT2, *client* market)
    * Intel® HD Graphics 520 (GT2, *client* market)
    * Intel® HD Graphics 530 (GT2, *client* market)
    * Intel® Iris® Graphics 540 (GT3e, *client* market)
    * Intel® Iris® Graphics 550 (GT3e, *client* market)
    * Intel® Iris® Pro Graphics 580 (GT4e, *client* market)
    * Intel® HD Graphics P530 (GT2, *server* market)
    * Intel® Iris® Pro Graphics P555 (GT3e, *server* market)
    * Intel® Iris® Pro Graphics P580 (GT4e, *server* market)
- Codename *Apollolake*:
    * Intel® HD Graphics 500
    * Intel® HD Graphics 505
- Codename *Kabylake*:
    * Intel® HD Graphics 610 (GT1, *client* market)
	* Intel® HD Graphics 615 (GT2, *client* market)
    * Intel® HD Graphics 620 (GT2, *client* market)
	* Intel® HD Graphics 630 (GT2, *client* market)
    * Intel® Iris® Graphics 640 (GT3e, *client* market)
    * Intel® Iris® Graphics 650 (GT3e, *client* market)
    * Intel® HD Graphics P630 (GT2, *server* market)
    * Intel® Iris® Pro Graphics 630 (GT2, *server* market)

clDNN currently uses OpenCL™ with multiple Intel® OpenCL™ extensions and requires Intel® Graphics Driver to run.

clDNN requires CPU with Intel® SSE/Intel® AVX support.

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
- *(optional)* [Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) 1.8.13 or later
    Needed for manual generation of documentation from inline comments or running `docs` custom target which will generate it automatically.

> [GraphViz\*](http://www.graphviz.org/Download..php) (2.38 or later) is also recommended to generate documentation with all embedded diagrams.
(Make sure that `dot` application is visible in the `PATH` environment variable.)

---

- The software was validated on:
    * CentOS* 7.2 with GNU* Compiler Collection 5.2 (64-bit only), using [Intel® Graphics Compute Runtime for OpenCL(TM)](https://software.intel.com/en-us/articles/opencl-drivers) .
    * Windows® 10 and Windows® Server 2012 R2 with MSVC 14.0, using [Intel® Graphics Driver for Windows* [24.20] driver package](https://downloadcenter.intel.com/download/27803/Graphics-Intel-Graphics-Driver-for-Windows-10?v=t).

	More information on Intel® OpenCL™ drivers can be found [here](https://software.intel.com/en-us/articles/opencl-drivers).

We recommend to use latest for Linux [link](https://github.com/intel/compute-runtime/releases) and 24.20 driver for Windows [link](https://downloadcenter.intel.com/download/27803/Graphics-Intel-Graphics-Driver-for-Windows-10?v=t).

## Installation

### Building

Download [clDNN source code](https://github.com/01org/cldnn/archive/master.zip)
or clone the repository to your system:

```
    git clone  https://github.com/intel/cldnn.git
```

Satisfy all software dependencies and ensure that the versions are correct before building.

clDNN uses multiple 3<sup>rd</sup>-party components. They are stored in binary form in `common` subdirectory. Currently they are prepared for MSVC++ and GCC\*. They will be cloned with repository.

---

clDNN uses a CMake-based build system. You can use CMake command-line tool or CMake GUI (`cmake-gui`) to generate required solution.
For Windows system, you can call in `cmd` (or `powershell`):
```shellscript
    @REM Generate 32-bit solution (solution contains multiple build configurations)...
    cmake -E make_directory build && cd build && cmake -G "Visual Studio 14 2015" ..
    @REM Generate 64-bit solution (solution contains multiple build configurations)...
    cmake -E make_directory build && cd build && cmake -G "Visual Studio 14 2015 Win64" ..
```
Created solution can be opened in Visual Studio 2015 or built using appropriate `msbuild` tool
(you can also use `cmake --build .` to select build tool automatically).

For Unix and Linux systems:
```shellscript
    @REM Create GNU makefile for release clDNN and build it...
    cmake -E make_directory build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make
    @REM Create Ninja makefile for debug clDNN and build it...
    cmake -E make_directory build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Debug .. && ninja -k 20
```

You can call also scripts in main directory of project which will create solutions/makefiles for clDNN (they
will generate solutions/makefiles in `build` subdirectory and binary outputs will be written to `build/out` subdirectory):
- `create_msvc_mscc.bat` (Windows\*, Visual Studio\* 2015)
- `create_unixmake_gcc.sh [Y|N] [<devtoolset-version>]` (Linux\*, GNU\* or Ninja\* makefiles, optional devtoolset support)
    * If you specify the first parameter as `Y`, the Ninja makefiles will be generated.
    * If you specify second parameter (number), the CMake will be called via `scl` with selected `devtoolset` version.

CMake solution offers multiple options which you can specify using normal CMake syntax (`-D<option-name>=<value>`):

| CMake option                              | Type     | Description                                                                  |
|:------------------------------------------|:---------|:-----------------------------------------------------------------------------|
| CMAKE\_BUILD\_TYPE                        | STRING   | Build configuration that will be used by generated makefiles (it does not affect multi-configuration generators like generators for Visual Studio solutions). Currently supported: `Debug` (default), `Release` |
| CMAKE\_INSTALL\_PREFIX                    | PATH     | Install directory prefix.                                                    |
| CLDNN\_\_ARCHITECTURE\_TARGET             | STRING   | Architecture of target system (where binary output will be deployed). CMake will try to detect it automatically (based on selected generator type, host OS and compiler properties). Specify this option only if CMake has problem with detection. Currently supported: `Windows32`, `Windows64`, `Linux64` |
| CLDNN\_\_OUTPUT\_DIR (CLDNN\_\_OUTPUT\_BIN\_DIR, CLDNN\_\_OUTPUT\_LIB\_DIR) | PATH | Location where built artifacts will be written to. It is set automatically to roughly `build/out/<arch-target>/<build-type>` subdirectory. For more control use: `CLDNN__OUTPUT_LIB_DIR` (specifies output path for static libraries) or `CLDNN__OUTPUT_BIN_DIR` (for shared libs and executables). |
|                                           |          |                                                                              |
| **CMake advanced option**                 | **Type** | **Description**                                                              |
| PYTHON\_EXECUTABLE                        | FILEPATH | Path to Python interpreter. CMake will try to detect Python. Specify this option only if CMake has problem with locating Python. |
| CLDNN\_\_IOCL\_ICD\_USE\_EXTERNAL         | BOOL     | Use this option to enable use of external Intel® OpenCL™ SDK as a source for ICD binaries and headers (based on `INTELOCLSDKROOT` environment variable). Default: `OFF` |
| CLDNN\_\_IOCL\_ICD\_VERSION               | STRING   | Version of Intel® OpenCL™ ICD binaries and headers to use (from `common` subdirectory). It is automatically setected by CMake (highest version). Specify, if you have multiple versions and want to use different than automatically selected. |
|                                           |          |                                                                              |
| CLDNN__COMPILE_LINK_ALLOW_UNSAFE_SIZE_OPT | BOOL     | Allow unsafe optimizations during linking (like aggressive dead code elimination, etc.). Default: `ON` |
| CLDNN__COMPILE_LINK_USE_STATIC_RUNTIME    | BOOL     | Link with static C++ runtime. Default: `OFF` (shared C++ runtime is used)    |
|                                           |          |                                                                              |
| CLDNN__INCLUDE_CORE                       | BOOL     | Include core clDNN library project in generated makefiles/solutions. Default: `ON` |
| CLDNN__INCLUDE_TESTS                      | BOOL     | Include tests application project (based on googletest framework) in generated makefiles/solutions . Default: `ON` |
|                                           |          |                                                                              |
| CLDNN__RUN_TESTS                          | BOOL     | Run tests after building `tests` project. This option requires `CLDNN__INCLUDE_TESTS` option to be `ON`. Default: `OFF` |
|                                           |          |                                                                              |
| CLDNN__CMAKE_DEBUG                        | BOOL     | Enable extended debug messages in CMake. Default: `OFF`                      |

---

clDNN includes unit tests implemented using the googletest framework. To validate your build, run `tests` target, e.g.:

```
    make tests
```

(Make sure that both `CLDNN__INCLUDE_TESTS` and `CLDNN__RUN_TESTS` were set to `ON` when invoking CMake.)

### Generating documentation

Documentation is provided inline and can be generated in HTML format with Doxygen. We recommend to use latest
[Doxygen\*](http://www.stack.nl/~dimitri/doxygen/download.html) and [GraphViz\*](http://www.graphviz.org/Download..php).

Documentation templates and configuration files are stored in `docs` subdirectory. You can simply call:

```shellscript
    cd docs && doxygen
```
to generate HTML documentation in `docs/html` subdirectory.

There is also custom CMake target named `docs` which will generate documentation in `CLDNN__OUTPUT_BIN_DIR/html` directory. For example, when using Unix makefiles, you can run:
```
    make docs
```
in order to create it.

### Deployment

Special `install` target will place the API header files and libraries in `/usr/local`
(`C:/Program Files/clDNN` or `C:/Program Files (x86)/clDNN` on Windows). To change
the installation path, use the option `-DCMAKE_INSTALL_PREFIX=<prefix>` when invoking CMake.

---


\* Other names and brands may be claimed as the property of others.

Copyright © 2017, Intel® Corporation

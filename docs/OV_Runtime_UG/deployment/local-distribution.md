# Local distribution {#openvino_docs_deploy_local_distribution}

The local distribution implies that each application or installer will have its own copies of OpenVINO Runtime binaries. For this, it's important to understand which minimal set of libraries is needed to deploy the application. Since the OpenVINO has quite scalable plugin architecture, let us go step by step to understand what is actually needed and what is optional.

> **NOTE**: the steps below are operation system independent and refer to a library file name without any prefixes (like `lib` on Unix systems) or suffixes (like `.dll` on Windows OS). **IMPORTANT** do not put `.lib` files on Windows OS to the distribution, because such files are needed only on build / linker stage.

> **NOTE**: Local dsitribution type is also appropriate for OpenVINO binaries built from sources using [Build instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build), but the guide below supposed OpenVINO Runtime is built dynamically. For case of [Static OpenVINO Runtime](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) select the required OpenVINO capabilities on CMake configuration stage using [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation), the build and link the OpenVINO components into the final application.

### C++ or C language

Independently on language used to write the application, `openvino` must always be put to the final distribution since is a core library which orshectrates with all the inference and frontend plugins.
If your application is written with C language, then you need to put `openvino_c` additionally.

The `plugins.xml` file with information about inference devices must also be taken as support file for `openvino`.

> **NOTE**: in Intel Distribution of OpenVINO, `openvino` depends on TBB libraries which are used by OpenVINO Runtime to optimally parallize the computations, so it must be put to the distribution package

### Compute devices

For each inference device, OpenVINO Runtime has its own plugin library:
- `openvino_intel_cpu_plugin` for [Intel CPU devices](../supported_plugins/CPU.md)
- `openvino_intel_gpu_plugin` for [Intel GPU devices](../supported_plugins/GPU.md)
 - Has `OpenCL` library as a dependency
- `openvino_intel_gna_plugin` for [Intel GNA devices](../supported_plugins/GNA.md)
 - Has `gna` backend library as a dependency
- `openvino_intel_myriad_plugin` for [Intel MYRIAD devices](../supported_plugins/MYRIAD.md)
 - Has `usb` library as a dependency
- `openvino_intel_hddl_plugin` for [Intel HDDL device](../supported_plugins/HDDL.md)
 - Has libraries from `runtime/3rdparty/hddl` as a dependency
- `openvino_arm_cpu_plugin` for [ARM CPU devices](../supported_plugins/ARM_CPU.md)

Depending on what device is used in the app, put the appropriate libraries to the distribution package.

### Execution capabilities

`HETERO`, `MULTI`, `BATCH`, `AUTO` execution capabilities can also be used explicitly or implicitly by the application. Use the following recommendation scheme to decide whether to put the appropriate libraries to the distribution package:
- If `AUTO` is used explicitly in the application or `ov::Core::compile_model` was called without device specifying, put the `openvino_auto_plugin` to the distribution
- If `MULTI` is used explicitly, put the `openvino_auto_plugin` to the distribution
- If `HETERO` is either used explicitly or `ov::hint::performance_mode` is used with GPU, put the `openvino_hetero_plugin` to the distribution
- If `BATCH` is either used explicitly or `ov::hint::performance_mode` is used with GPU, put the `openvino_batch_plugin` to the distribution

### Reading models

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:
- To read OpenVINO IR `openvino_ir_frontend` is used
- To read ONNX file format `openvino_onnx_frontend` is used
- To read Paddle file format `openvino_paddle_frontend` is used

Depending on what types of model file format are used in the application, peek up the appropriate libraries.

> **NOTE**: the recommended way to optimize the size of final distribution page is to [convert models using Model Optimizer](../../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to OpenVINO IR, in this case you don't have to keep ONNX, Paddle and other frontend libraries in the distribution package.

### (Legacy) Preprocessing via G-API

> **NOTE**: [G-API](../../gapi/gapi_intro.md) preprocessing is a legacy functionality, use [preprocessing capabilities from OpenVINO 2.0](../preprocessing_overview.md) which do not require any additional specific libraries.

If the application uses `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` methods, OpenVINO Runtime dynamically loads `openvino_gapi_preproc` plugin to perform preprocessing via G-API.

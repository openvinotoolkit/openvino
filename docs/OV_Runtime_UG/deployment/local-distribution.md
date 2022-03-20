# Local distribution {#openvino_docs_deploy_local_distribution}

The local distribution implies that each C or C++ application / installer will have its own copies of OpenVINO Runtime binaries. However, OpenVINO has a scalable plugin-based architecture which implies that some components can be loaded in runtime only if they are really needed. So, it is important to understand which minimal set of libraries is really needed to deploy the application and this guide helps to achieve this goal.

> **NOTE**: The steps below are operation system independent and refer to a library file name without any prefixes (like `lib` on Unix systems) or suffixes (like `.dll` on Windows OS). Do not put `.lib` files on Windows OS to the distribution, because such files are needed only on build / linker stage.

Local dsitribution is also appropriate for OpenVINO binaries built from sources using [Build instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build), but the guide below supposes OpenVINO Runtime is built dynamically. For case of [Static OpenVINO Runtime](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) select the required OpenVINO capabilities on CMake configuration stage using [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation), the build and link the OpenVINO components into the final application.

### C++ or C language

Independently on language used to write the application, `openvino` must always be put to the final distribution since is a core library which orshectrates with all the inference and frontend plugins.
If your application is written with C language, then you need to put `openvino_c` additionally.

The `plugins.xml` file with information about inference devices must also be taken as support file for `openvino`.

> **NOTE**: in Intel Distribution of OpenVINO, `openvino` depends on TBB libraries which are used by OpenVINO Runtime to optimally saturate the devices with computations, so it must be put to the distribution package

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

Depending on what devices is used in the app, put the appropriate libraries to the distribution package.

### Execution capabilities

`HETERO`, `MULTI`, `BATCH`, `AUTO` execution capabilities can also be used explicitly or implicitly by the application. Use the following recommendation scheme to decide whether to put the appropriate libraries to the distribution package:
- If [AUTO](../auto_device_selection.md) is used explicitly in the application or `ov::Core::compile_model` is used without specifying a device, put the `openvino_auto_plugin` to the distribution
 > **NOTE**: Auto device selection relies on [inference device plugins](../supported_plugins/Device_Plugins.md), so if are not sure what inference devices are available on target machine, put all inference plugin libraries to the distribution. If the `ov::device::priorities` is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If [MULTI](../multi_device.md) is used explicitly, put the `openvino_auto_plugin` to the distribution
- If [HETERO](../hetero_execution.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, put the `openvino_hetero_plugin` to the distribution
- If [BATCH](../automatic_batching.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, put the `openvino_batch_plugin` to the distribution

### Reading models

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:
- To read OpenVINO IR `openvino_ir_frontend` is used
- To read ONNX file format `openvino_onnx_frontend` is used
- To read Paddle file format `openvino_paddle_frontend` is used

Depending on what types of model file format are used in the application in `ov::Core::read_model`, peek up the appropriate libraries.

> **NOTE**: The recommended way to optimize the size of final distribution package is to [convert models using Model Optimizer](../../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to OpenVINO IR, in this case you don't have to keep ONNX, Paddle and other frontend libraries in the distribution package.

### (Legacy) Preprocessing via G-API

> **NOTE**: [G-API](../../gapi/gapi_intro.md) preprocessing is a legacy functionality, use [preprocessing capabilities from OpenVINO 2.0](../preprocessing_overview.md) which do not require any additional libraries.

If the application uses `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` methods, OpenVINO Runtime dynamically loads `openvino_gapi_preproc` plugin to perform preprocessing via G-API.

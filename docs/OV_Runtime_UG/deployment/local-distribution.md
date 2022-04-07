# Local Distribution {#openvino_docs_deploy_local_distribution}

The local distribution implies that each C or C++ application / installer will have its own copies of OpenVINO Runtime binaries. However, OpenVINO has a scalable plugin-based architecture, which implies that some components can be loaded in runtime only if they are really needed. Therefore, it is important to understand which minimal set of libraries is really needed to deploy the application. This guide will help you achieve this goal.

> **NOTE**: The steps below are operation system independent and refer to a library file name without any prefixes (like `lib` on Unix systems) or suffixes (like `.dll` on Windows OS). Do not include `.lib` files on Windows OS in the distribution, because such files are needed only on a linker stage.

Local distribution is also appropriate for OpenVINO binaries built from sources using the [Build instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build), but the guide below supposes OpenVINO Runtime is built dynamically. For [Static OpenVINO Runtime](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) select the required OpenVINO capabilities on CMake configuration stage, using [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation) and the build, and link the OpenVINO components into the final application.

### C++ or C Language

Regardless of the language used to write the application, `openvino` must always be included in the final distribution since it is a core library, which orchestrates with all the inference and frontend plugins.
If your application is written with C language, then you need to include `openvino_c` additionally.

The `plugins.xml` file with information about inference devices must also be taken as support file for `openvino`.

> **NOTE**: In Intel Distribution of OpenVINO, `openvino` depends on TBB libraries, which are used by OpenVINO Runtime to optimally saturate the devices with computations, so it must be included in the distribution package.

### Pluggable Components

The picture below demonstrates dependencies between the OpenVINO Runtime core and pluggable libraries:

![deployment_full]

#### Compute Devices

For each inference device, OpenVINO Runtime has its own plugin library:
- `openvino_intel_cpu_plugin` for [Intel CPU devices](../supported_plugins/CPU.md)
- `openvino_intel_gpu_plugin` for [Intel GPU devices](../supported_plugins/GPU.md)
- `openvino_intel_gna_plugin` for [Intel GNA devices](../supported_plugins/GNA.md)
- `openvino_intel_myriad_plugin` for [Intel MYRIAD devices](../supported_plugins/MYRIAD.md)
- `openvino_intel_hddl_plugin` for [Intel HDDL device](../supported_plugins/HDDL.md)
- `openvino_arm_cpu_plugin` for [ARM CPU devices](../supported_plugins/ARM_CPU.md)

Depending on what devices are used in the app, include the appropriate libraries into the distribution package.

As shown in the picture above, some plugin libraries may have OS-specific dependencies which are either backend libraries or additional supports files with firmware, etc. Refer to the table below for details:

@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="Windows OS: Click to expand/collapse">

@endsphinxdirective

| Device      | Dependency |
|-------------|------------|
| CPU         |  `-`       |
| GPU         | `OpenCL.dll`, `cache.json` |
| MYRIAD      | `usb.dll`, `usb-ma2x8x.mvcmd`, `pcie-ma2x8x.elf` |
| HDDL        | `bsl.dll`, `hddlapi.dll`, `json-c.dll`, `libcrypto-1_1-x64.dll`, `libssl-1_1-x64.dll`, `mvnc-hddl.dll` |
| GNA         | `gna.dll` |
| Arm® CPU    |  `-`      |

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective
@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="Linux OS: Click to expand/collapse">

@endsphinxdirective

| Device      | Dependency  |
|-------------|-------------|
| CPU         |  `-`        |
| GPU         | `libOpenCL.so`, `cache.json` |
| MYRIAD      | `libusb.so`, `usb-ma2x8x.mvcmd`, `pcie-ma2x8x.mvcmd` |
| HDDL        | `libbsl.so`, `libhddlapi.so`, `libmvnc-hddl.so` |
| GNA         | `gna.dll`   |
| Arm® CPU    |  `-`        |

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective
@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="MacOS: Click to expand/collapse">

@endsphinxdirective

| Device      | Dependency  |
|-------------|-------------|
| CPU         |     `-`     |
| MYRIAD      | `libusb.dylib`, `usb-ma2x8x.mvcmd`, `pcie-ma2x8x.mvcmd` |
| Arm® CPU    |  `-`        |

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective

#### Execution Capabilities

`HETERO`, `MULTI`, `BATCH`, `AUTO` execution capabilities can also be used explicitly or implicitly by the application. Use the following recommendation scheme to decide whether to include the appropriate libraries in the distribution package:
- If [AUTO](../auto_device_selection.md) is used explicitly in the application or `ov::Core::compile_model` is used without specifying a device, include the `openvino_auto_plugin` in the distribution
 > **NOTE**: Auto device selection relies on [inference device plugins](../supported_plugins/Device_Plugins.md), so if are not sure what inference devices are available on target machine, include all inference plugin libraries in the distribution. If the `ov::device::priorities` is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If [MULTI](../multi_device.md) is used explicitly, include the `openvino_auto_plugin` in the distribution
- If [HETERO](../hetero_execution.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, include the `openvino_hetero_plugin` in the distribution
- If [BATCH](../automatic_batching.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, include the `openvino_batch_plugin` in the distribution

#### Reading Models

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:
- To read OpenVINO IR `openvino_ir_frontend` is used
- To read ONNX file format `openvino_onnx_frontend` is used
- To read Paddle file format `openvino_paddle_frontend` is used

Depending on what types of model file formats are used in the application in `ov::Core::read_model`, peek up the appropriate libraries.

> **NOTE**: The recommended way to optimize the size of final distribution package is to [convert models using Model Optimizer](../../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md) to OpenVINO IR, in this case you do not have to keep ONNX, Paddle and other frontend libraries in the distribution package.

#### (Legacy) Preprocessing via G-API

> **NOTE**: [G-API](../../gapi/gapi_intro.md) preprocessing is a legacy functionality, use [preprocessing capabilities from OpenVINO 2.0](../preprocessing_overview.md) which do not require any additional libraries.

If the application uses `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` methods, OpenVINO Runtime dynamically loads `openvino_gapi_preproc` plugin to perform preprocessing via G-API.

### Examples

#### CPU + IR in C-written Application

C-written application performs inference on CPU and reads models stored as OpenVINO IR:
- `openvino_c` library is a main dependency of the application. It links against this library.
- `openvino` is used as a private dependency for `openvino` and is also used in the deployment.
- `openvino_intel_cpu_plugin` is used for inference.
- `openvino_ir_frontend` is used to read a source model.

#### MULTI Execution on GPU and MYRIAD in tput Mode

C++ written application performs inference [simultaneously on GPU and MYRIAD devices](../multi_device.md) with `ov::hint::PerformanceMode::THROUGHPUT` property. 
It reads models stored in ONNX file format:
- `openvino` library is a main dependency of the application. It links against this library.
- `openvino_intel_gpu_plugin` and `openvino_intel_myriad_plugin` are used for inference.
- `openvino_auto_plugin` is used for `MULTI` multi-device execution.
- `openvino_auto_batch_plugin` can also be included in the distribution to improve the saturation of [Intel GPU](../supported_plugins/GPU.md) device. If there is no such plugin, [Automatic batching](../automatic_batching.md) is turned off.
- `openvino_onnx_frontend` is used to read a source model.

#### Auto Device Selection between HDDL and CPU

C++ written application performs inference with [automatic device selection](../auto_device_selection.md), with device list limited to HDDL and CPU.
A model is [created using C++ code](../model_representation.md):
- `openvino` library is a main dependency of the application. It links against this library.
- `openvino_auto_plugin` is used to enable automatic device selection feature.
- `openvino_intel_hddl_plugin` and `openvino_intel_cpu_plugin` are used for inference, `AUTO` selects between CPU and HDDL devices according to their physical existance on deployed machine.
- No frontend library is needed because `ov::Model` is created in code.

[deployment_full]: ../../img/deployment_full.png

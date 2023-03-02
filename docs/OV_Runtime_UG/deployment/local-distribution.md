# Libraries for Local Distribution {#openvino_docs_deploy_local_distribution}

With a local distribution, each C or C++ application/installer will have its own copies of OpenVINO Runtime binaries. However, OpenVINO has a scalable plugin-based architecture, which means that some components can be loaded in runtime only when they are really needed. Therefore, it is important to understand which minimal set of libraries is really needed to deploy the application. This guide helps you to achieve that goal.


Local dsitribution is also appropriate for OpenVINO binaries built from sources using [Build instructions](https://github.com/openvinotoolkit/openvino/wiki#how-to-build), but the guide below supposes OpenVINO Runtime is built dynamically. For case of [Static OpenVINO Runtime](https://github.com/openvinotoolkit/openvino/wiki/StaticLibraries) select the required OpenVINO capabilities on CMake configuration stage using [CMake Options for Custom Compilation](https://github.com/openvinotoolkit/openvino/wiki/CMakeOptionsForCustomCompilation), the build and link the OpenVINO components into the final application.

> **NOTE**: The steps below are operating system independent and refer to a library file name without any prefixes (like `lib` on Unix systems) or suffixes (like `.dll` on Windows OS). Do not put `.lib` files on Windows OS to the distribution, because such files are needed only on a linker stage.

## Library Requirements for C++ and C Languages

Independent on the language used to write the application, the `openvino` library must always be put to the final distribution, since it's a core library which orchestrates with all the inference and frontend plugins. In Intel® Distribution of OpenVINO™ toolkit, `openvino` depends on the TBB libraries which are used by OpenVINO Runtime to optimally saturate the devices with computations, so it must be put to the distribution package.

If your application is written with C language, you need to put the `openvino_c` library additionally.

The `plugins.xml` file with information about inference devices must also be taken as a support file for `openvino`.


## Libraries for Pluggable Components

The picture below presents dependencies between the OpenVINO Runtime core and pluggable libraries:

![](../../img/deployment_full.svg)

### Libraries for Compute Devices

For each inference device, OpenVINO Runtime has its own plugin library:
- `openvino_intel_cpu_plugin` for [Intel® CPU devices](../supported_plugins/CPU.md).
- `openvino_intel_gpu_plugin` for [Intel® GPU devices](../supported_plugins/GPU.md).
- `openvino_intel_gna_plugin` for [Intel® GNA devices](../supported_plugins/GNA.md).
- `openvino_arm_cpu_plugin` for [ARM CPU devices](../supported_plugins/ARM_CPU.md).

Depending on what devices are used in the app, the appropriate libraries need to be put to the distribution package.

As it is shown on the picture above, some plugin libraries may have OS-specific dependencies which are either backend libraries or additional supports files with firmware, etc. Refer to the table below for details:

@sphinxdirective

.. raw:: html

    <div class="collapsible-section" data-title="Windows OS: Click to expand/collapse">

@endsphinxdirective

| Device      | Dependency |
|-------------|------------|
| CPU         |  `-`       |
| GPU         | `OpenCL.dll`, `cache.json` |
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
| Arm® CPU    |  `-`        |

@sphinxdirective

.. raw:: html

    </div>

@endsphinxdirective

### Libraries for Execution Modes

The `HETERO`, `MULTI`, `BATCH` and `AUTO` execution modes can also be used explicitly or implicitly by the application. Use the following recommendation scheme to decide whether to put the appropriate libraries to the distribution package:
- If [AUTO](../auto_device_selection.md) is used explicitly in the application or `ov::Core::compile_model` is used without specifying a device, put `openvino_auto_plugin` to the distribution.
  > **NOTE**: Automatic Device Selection relies on [inference device plugins](../supported_plugins/Device_Plugins.md). If you are not sure about what inference devices are available on target system, put all the inference plugin libraries to the distribution. If `ov::device::priorities` is used for `AUTO` to specify a limited device list, grab the corresponding device plugins only.

- If [MULTI](../multi_device.md) is used explicitly, put `openvino_auto_plugin` to the distribution.
- If [HETERO](../hetero_execution.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, put `openvino_hetero_plugin` to the distribution.
- If [BATCH](../automatic_batching.md) is either used explicitly or `ov::hint::performance_mode` is used with GPU, put `openvino_batch_plugin` to the distribution.

### Frontend Libraries for Reading Models

OpenVINO Runtime uses frontend libraries dynamically to read models in different formats:
- `openvino_ir_frontend` is used to read OpenVINO IR.
- `openvino_tensorflow_frontend` is used to read TensorFlow file format.
- `openvino_onnx_frontend` is used to read ONNX file format.
- `openvino_paddle_frontend` is used to read Paddle file format.

Depending on the model format types that are used in the application in `ov::Core::read_model`, pick up the appropriate libraries.

> **NOTE**: To optimize the size of final distribution package, you are recommended to convert models to OpenVINO IR by using [Model Optimizer](../../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md). This way you don't have to keep TensorFlow, ONNX, PaddlePaddle, and other frontend libraries in the distribution package.

### (Legacy) Preprocessing via G-API

> **NOTE**: [G-API](../../gapi/gapi_intro.md) preprocessing is a legacy functionality, use [preprocessing capabilities from OpenVINO 2.0](../preprocessing_overview.md) which do not require any additional libraries.

If the application uses `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` methods, OpenVINO Runtime dynamically loads `openvino_gapi_preproc` plugin to perform preprocessing via G-API.

## Examples

**CPU + OpenVINO IR in C application**

In this example, the application is written in C language, performs inference on CPU, and reads models stored as the OpenVINO IR format. The following libraries are used:
- The `openvino_c` library is a main dependency of the application. It links against this library.
- The `openvino` library is used as a private dependency for `openvino_c` and is also used in the deployment.
- `openvino_intel_cpu_plugin` is used for inference.
- `openvino_ir_frontend` is used to read source models.

**MULTI execution on GPU and CPU in `tput` mode**

In this example, the application is written in C++, performs inference [simultaneously on GPU and CPU devices](../multi_device.md) with the `ov::hint::PerformanceMode::THROUGHPUT` property set, and reads models stored in the ONNX format. The following libraries are used:
- The `openvino` library is a main dependency of the application. It links against this library.
- `openvino_intel_gpu_plugin` and `openvino_intel_cpu_plugin` are used for inference.
- `openvino_auto_plugin` is used for Multi-Device Execution.
- `openvino_auto_batch_plugin` can be also put to the distribution to improve the saturation of [Intel® GPU](../supported_plugins/GPU.md) device. If there is no such plugin, [Automatic Batching](../automatic_batching.md) is turned off.
- `openvino_onnx_frontend` is used to read source models.

**Auto-Device Selection between GPU and CPU**

In this example, the application is written in C++, performs inference with the [Automatic Device Selection](../auto_device_selection.md) mode, limiting device list to GPU and CPU, and reads models [created using C++ code](../model_representation.md). The following libraries are used:
- The `openvino` library is a main dependency of the application. It links against this library.
- `openvino_auto_plugin` is used to enable Automatic Device Selection.
- `openvino_intel_gpu_plugin` and `openvino_intel_cpu_plugin` are used for inference. AUTO selects between CPU and GPU devices according to their physical existence on the deployed machine.
- No frontend library is needed because `ov::Model` is created in code.

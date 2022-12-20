# 安装和部署 {#openvino_2_0_deployment_zh_CN}

OpenVINO™ API 2.0 的主要理念之一是“易于使用”，这包括：

* 从不同框架迁移到 OpenVINO™ 的简化。
* OpenVINO™ 的组织方式。
* 开发工具的使用。
* 开发和部署支持 OpenVINO™ 的应用。

为实现该目标，2022.1 版本的 OpenVINO™ 对安装和部署流程做出了重大更改。本指南将向您介绍这些更改。

## 安装包仅包含 OpenVINO™ 运行时

自 OpenVINO™ 2022.1 起，开发工具仅通过 [PyPI](https://pypi.org/project/openvino-dev/) 分发，而不再包含在 OpenVINO™ 安装包中。如需这些组件的列表，请参阅[安装概述](../../../install_guides/installing-openvino-overview.md)指南。这种方法的优势包括：

* 简化用户体验 - 在早期版本中，不同分发类型安装和使用 OpenVINO™ 开发工具的方式各不相同（OpenVINO™ 安装程序与 PyPI），
* 因此确保依赖项通过 PIP 程序包管理器进行正确处理，并支持开发工具的虚拟环境。

OpenVINO™ 2022.1 安装包的组织结构如下：

- `runtime` 文件夹包含标头、库和 CMake 接口。
- `tools` 文件夹包含[编译工具](../../compile_tool/README_zh_CN.md)、[部署管理器](../../../OV_Runtime_UG/deployment/deployment-manager-tool.md)和一组 `requirements.txt` 文件。其中提供了对应版本的 `openvino-dev` 程序包的链接。
- `python` 文件夹包含 Python 版本的 OpenVINO™ 运行时。

## 通过 PyPI 安装 OpenVINO™ 开发工具

由于 OpenVINO™ 开发工具不再位于安装包中。因此安装过程也已发生变化。本节将通过与早期版本进行比较来介绍整个安装过程。

### 对于 2022.1 之前的版本

在早期版本中，OpenVINO™ 开发工具是主程序包的一部分。安装程序包后，如欲转换模型（例如，TensorFlow），您需要通过使用 `requirements_tf.txt` 等要求文件安装其他依赖项，通过 `setup.py` 脚本安装训练后优化工具和精度检查器工具，然后使用 `setupvars` 脚本将工具提供给下列命令：

```sh
$ mo.py -h
```

### 对于 2022.1 及更高版本

在 OpenVINO™ 2022.1 及更高版本中，您只能从 [PyPI](https://pypi.org/project/openvino-dev/) 存储库使用以下命令（例如，TensorFlow）安装开发工具：

```sh
$ python3 -m pip install -r <INSTALL_DIR>/tools/requirements_tf.txt 
```

这将通过 `openvino-dev` 程序包安装所有开发工具和支持 TensorFlow 所需的其他组件（请参阅**步骤 4：安装程序包** - 位于 [PyPI 页面](https://pypi.org/project/openvino-dev/)，了解其他框架的参数）。

然后，相关命令即可使用这些工具，例如：

```sh
$ mo -h
$ pot -h
```

不需要安装任何其他依赖项。有关安装步骤的更多详细信息，请参阅[安装 OpenVINO™ 开发工具](../../../install_guides/installing-model-dev-tools.md)。

## 构建 C/C++ 应用的接口发生更改

支持 API 2.0 的全新 OpenVINO™ 运行时还在构建 C/C++ 应用方面做出了一些更改。

### CMake 接口

对 CMake 接口的更改如下：

**使用早期版本的推理引擎**：

```cmake
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
add_executable(ie_ngraph_app main.cpp)
target_link_libraries(ie_ngraph_app PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
```

**使用 OpenVINO™ 运行时 2022.1 (API 2.0)**：

```cmake
find_package(OpenVINO REQUIRED)
add_executable(ov_app main.cpp)
target_link_libraries(ov_app PRIVATE openvino::runtime)

add_executable(ov_c_app main.c)
target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)
```

### 原生接口

通过使用 MSVC IDE、UNIX 生成文件和如下所示已做出更改的任何其他接口，在不借助 CMake 接口的情况下，可以构建应用：

**使用早期版本的推理引擎**：

@sphinxdirective

.. tab:: Include dirs

  .. code-block:: sh
    
    <INSTALL_DIR>/deployment_tools/inference_engine/include
    <INSTALL_DIR>/deployment_tools/ngraph/include

.. tab:: Path to libs

  .. code-block:: sh

    <INSTALL_DIR>/deployment_tools/inference_engine/lib/intel64/Release
    <INSTALL_DIR>/deployment_tools/ngraph/lib/

.. tab:: Shared libs

  .. code-block:: sh

    // UNIX systems
    inference_engine.so ngraph.so

    // Windows
    inference_engine.dll ngraph.dll

.. tab:: (Windows) .lib files

  .. code-block:: sh
  
    ngraph.lib
    inference_engine.lib

@endsphinxdirective

**使用 OpenVINO™ 运行时 2022.1 (API 2.0)**：

@sphinxdirective

.. tab:: Include dirs

  .. code-block:: sh

    <INSTALL_DIR>/runtime/include

.. tab:: Path to libs

  .. code-block:: sh

    <INSTALL_DIR>/runtime/lib/intel64/Release

.. tab:: Shared libs

  .. code-block:: sh

    // UNIX systems
    openvino.so

    // Windows
    openvino.dll

.. tab:: (Windows) .lib files

  .. code-block:: sh

    openvino.lib

@endsphinxdirective

## 库结构更清晰，部署更便捷

OpenVINO™ 2022.1 对库进行了重组，以使部署更加方便。在早期版本中，需要使用几个库才能执行部署步骤。现在，根据您的开发语言，您只需使用 `openvino` 或 `openvino_c`，并借助必要的插件即可完成任务。例如，`openvino_intel_cpu_plugin` 和 `openvino_ir_frontend` 插件将帮助加载 OpenVINO™ IR 并在 CPU 设备上执行接口（有关更多详细信息，请参阅[通过 OpenVINO™ 进行本地分发](../../../OV_Runtime_UG/deployment/local-distribution.md)）。

下面对 OpenVINO™ 2022.1 与早期版本之间的库结构进行了详细比较：

* 从 2022.1 版本开始，将使用包含所有功能的单一核心库（`openvino` 用于 C++ 运行时、`openvino_c` 用于推理引擎 API C 接口），而不使用之前包含 `inference_engine`、`ngraph`、`inference_engine_transformations` 和 `inference_engine_lp_transformations` 的核心库。
* 可选的 `inference_engine_preproc` 预处理库（如果使用 `InferenceEngine::PreProcessInfo::setColorFormat` 或 `InferenceEngine::PreProcessInfo::setResizeAlgorithm`）已重命名为 `openvino_gapi_preproc`，并已在 2022.1 中弃用。有关更多详细信息，请参阅 [OpenVINO™ API 2.0 的预处理功能](preprocessing_zh_CN.md)。
* 插件库已重命名，如下所述：
   * `openvino_intel_cpu_plugin` 用于 [CPU](../supported_plugins/CPU_zh_CN.md) 设备，而不使用 `MKLDNNPlugin`。
   * `openvino_intel_gpu_plugin` 用于 [GPU](../supported_plugins/GPU_zh_CN.md) 设备，而不使用 `clDNNPlugin`。
   * `openvino_auto_plugin` 用于[自动设备插件](../../../OV_Runtime_UG/auto_device_selection.md)。
* 用于读取和转换模型的插件已更改如下：
   * `openvino_ir_frontend` 用于读取 IR，而不使用 `inference_engine_ir_reader`。
   * `openvino_onnx_frontend` 用于读取 ONNX 模型，而不使用 `inference_engine_onnx_reader`（包括其依赖项）。
   * `openvino_paddle_frontend` 已添加到 2022.1 中，用于读取 PaddlePaddle 模型。

<!-----
Older versions of OpenVINO had several core libraries and plugin modules:
- Core: `inference_engine`, `ngraph`, `inference_engine_transformations`, `inference_engine_lp_transformations`
- Optional `inference_engine_preproc` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` are used)
- Plugin libraries:
 - `MKLDNNPlugin` for [CPU](../supported_plugins/CPU_zh_CN.md) device
 - `clDNNPlugin` for [GPU](../supported_plugins/GPU_zh_CN.md) device
 - `MultiDevicePlugin` for [Multi-device execution](../multi_device.md)
 - others
- Plugins to read and convert a model:
 - `inference_engine_ir_reader` to read OpenVINO IR
 - `inference_engine_onnx_reader` (with its dependencies) to read ONNX models
Now, the modularity is more clear:
- A single core library with all the functionality `openvino` for C++ runtime
- `openvino_c` with Inference Engine API C interface
- **Deprecated** Optional `openvino_gapi_preproc` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` are used)
 - Use [preprocessing capabilities from OpenVINO 2.0](../preprocessing_overview.md)
- Plugin libraries with clear names:
 - `openvino_intel_cpu_plugin`
 - `openvino_intel_gpu_plugin`
 - `openvino_auto_plugin`
 - others
- Plugins to read and convert models:
 - `openvino_ir_frontend` to read OpenVINO IR
 - `openvino_onnx_frontend` to read ONNX models
 - `openvino_paddle_frontend` to read Paddle models
---->

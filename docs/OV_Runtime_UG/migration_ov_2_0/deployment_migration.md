# Installation & Deployment {#openvino_2_0_deployment}

"Easy to use" is one of the main concepts for OpenVINO™ API 2.0. It includes not only simplifying the migration from frameworks to OpenVINO, but also how OpenVINO is organized, how the development tools are used, and how to develop and deploy OpenVINO-based applications.

To accomplish that, we have made some changes on the installation and deployment of OpenVINO in the 2022.1 release. This guide will walk you through them.

## Installer Package Contains OpenVINO™ Runtime Only

Starting from OpenVINO 2022.1, Model Optimizer, Post-Training Optimization tool and Python-based Development tools such as Open Model Zoo tools are distributed via [PyPI](https://pypi.org/project/openvino-dev/) only, and are no longer included in the OpenVINO installer package. This change has several benefits as it:

* Simplifies the user experience. In previous versions, the installation and usage of OpenVINO Development Tools differ according to the distribution type (via an OpenVINO installer or PyPI). 
* Ensures that dependencies are handled properly via the PIP package manager and support virtual environments of development tools.

The structure of OpenVINO 2022.1 installer package has been organized as below:

- The `runtime` folder includes headers, libraries and CMake interfaces.
- The `tools` folder contains [the compile tool](../../../tools/compile_tool/README.md), [deployment manager](../../install_guides/deployment-manager-tool.md) and a set of `requirements.txt` files with links to the corresponding versions of the `openvino-dev` package.
- The `python` folder contains the Python version for OpenVINO Runtime.

## Installing OpenVINO Development Tools via PyPI

Since OpenVINO Development Tools is no longer in the installer package, the installation process has changed too. This section describes it through a comparison with previous versions.

### For Versions Prior to 2022.1

In previous versions, OpenVINO Development Tools is a part of main package. After the package is installed, to convert models (for example, TensorFlow), you need to install additional dependencies by using the requirements files such as `requirements_tf.txt`, install Post-Training Optimization tool and Accuracy Checker tool via the `setup.py` scripts, and then use the `setupvars` scripts to make the tools available to the following command:

```sh
$ mo.py -h
```

### For 2022.1 and After

Starting from OpenVINO 2022.1, you can install the development tools from [PyPI](https://pypi.org/project/openvino-dev/) repository only, using the following command (taking TensorFlow as an example):

```sh
$ python3 -m pip install -r <INSTALL_DIR>/tools/requirements_tf.txt 
```

This will install all the development tools and additional necessary components to work with TensorFlow via the `openvino-dev` package (see **Step 4. Install the Package** on the [PyPI page](https://pypi.org/project/openvino-dev/) for parameters of other frameworks).

Then, the tools can be used by commands like:

```sh
$ mo -h
$ pot -h
```

You don't have to install any other dependencies. For more details on the installation steps, see [Install OpenVINO Development Tools](../../install_guides/installing-model-dev-tools.md).

## Interface Changes for Building C/C++ Applications

The new OpenVINO Runtime with API 2.0 has also brought some changes for builiding your C/C++ applications.

### CMake Interface

The CMake interface has been changed as below:

**With Inference Engine of previous versions**:

```cmake
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
add_executable(ie_ngraph_app main.cpp)
target_link_libraries(ie_ngraph_app PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
```

**With OpenVINO Runtime 2022.1 (API 2.0)**:

```cmake
find_package(OpenVINO REQUIRED)
add_executable(ov_app main.cpp)
target_link_libraries(ov_app PRIVATE openvino::runtime)

add_executable(ov_c_app main.c)
target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)
```

### Native Interfaces

To build applications without CMake interface, you can also use MSVC IDE, UNIX makefiles and any other interfaces, which have been changed as below:

**With Inference Engine of previous versions**:

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

**With OpenVINO Runtime 2022.1 (API 2.0)**:

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

## Clearer Library Structure for Deployment

OpenVINO 2022.1 has reorganized the libraries to make it easier for deployment. In previous versions, to perform deployment steps, you have to use several libraries. Now you can just use `openvino` or `openvino_c` based on your developing language plus necessary plugins to complete your task. For example, `openvino_intel_cpu_plugin` and `openvino_ir_frontend` plugins will enable you to load OpenVINO IRs and perform inference on CPU device.

Here you can find some detailed comparisons on library structure between OpenVINO 2022.1 and previous versions:

* A single core library with all the functionalities (`openvino` for C++ Runtime, `openvino_c` for Inference Engine API C interface) is used in 2022.1, instead of the previous core libraries which contain `inference_engine`, `ngraph`, `inference_engine_transformations` and `inference_engine_lp_transformations`.
* The optional `inference_engine_preproc` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` is used) is deprecated in 2022.1. See more details on [Preprocessing capabilities of OpenVINO API 2.0](preprocessing.md).
* The libraries of plugins are renamed as below:
   * `openvino_intel_cpu_plugin` is used for [CPU](../supported_plugins/CPU.md) device instead of `MKLDNNPlugin` in previous versions.
   * `openvino_intel_gpu_plugin` is used for [GPU](../supported_plugins/GPU.md) device instead of `clDNNPlugin` in previous versions.
   * `openvino_auto_plugin` is used for [Auto-Device Plugin](../auto_device_selection.md) in 2022.1.
* The plugins for reading and converting models have been changed as below:
   * `openvino_ir_frontend` is used to read IRs instead of `inference_engine_ir_reader` in previous versions.
   * `openvino_onnx_frontend` is used to read ONNX models instead of `inference_engine_onnx_reader` (with its dependencies) in previous versions. 
   * `openvino_paddle_frontend` is added in 2022.1 to read PaddlePaddle models.

<!-----
Older versions of OpenVINO had several core libraries and plugin modules:
- Core: `inference_engine`, `ngraph`, `inference_engine_transformations`, `inference_engine_lp_transformations`
- Optional `inference_engine_preproc` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` are used)
- Plugin libraries:
 - `MKLDNNPlugin` for [CPU](../supported_plugins/CPU.md) device
 - `clDNNPlugin` for [GPU](../supported_plugins/GPU.md) device
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
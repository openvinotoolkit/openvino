# Installation & Deployment {#openvino_2_0_deployment}

One of the main concepts for OpenVINO™ API 2.0 is being "easy to use", which includes:
* Simplification of migration from different frameworks to OpenVINO.
* Organization of OpenVINO. 
* Usage of development tools.
* Development and deployment of OpenVINO-based applications.

To accomplish that, the 2022.1 release OpenVINO introduced significant changes to the installation and deployment processes. This guide will walk you through these changes.

## The Installer Package Contains OpenVINO™ Runtime Only

Since OpenVINO 2022.1, development tools have been distributed only via [PyPI](https://pypi.org/project/openvino-dev/), and are no longer included in the OpenVINO installer package. For a list of these components, refer to the [installation overview](../../../install_guides/installing-openvino-overview.md) guide. Benefits of this approach include:

* simplification of the user experience - in previous versions, installation and usage of OpenVINO Development Tools differed from one distribution type to another (the OpenVINO installer vs. PyPI), 
* ensuring that dependencies are handled properly via the PIP package manager, and support virtual environments of development tools.

The structure of the OpenVINO 2022.1 installer package has been organized as follows:

- The `runtime` folder includes headers, libraries and CMake interfaces.
- The `tools` folder contains [the compile tool](../../../tools/compile_tool/README.md), [deployment manager](../../OV_Runtime_UG/deployment/deployment-manager-tool.md), and a set of `requirements.txt` files with links to the corresponding versions of the `openvino-dev` package.
- The `python` folder contains the Python version for OpenVINO Runtime.

## Installing OpenVINO Development Tools via PyPI

Since OpenVINO Development Tools is no longer in the installer package, the installation process has also changed. This section describes it through a comparison with previous versions.

### For Versions Prior to 2022.1

In previous versions, OpenVINO Development Tools was a part of the main package. After the package was installed, to convert models (for example, TensorFlow), you needed to install additional dependencies by using the requirement files, such as `requirements_tf.txt`, install Post-Training Optimization tool and Accuracy Checker tool via the `setup.py` scripts, and then use the `setupvars` scripts to make the tools available to the following command:

```sh
$ mo.py -h
```

### For 2022.1 and After

In OpenVINO 2022.1 and later, you can install the development tools only from a [PyPI](https://pypi.org/project/openvino-dev/) repository, using the following command (taking TensorFlow as an example):

```sh
$ python3 -m pip install -r <INSTALL_DIR>/tools/requirements_tf.txt 
```

This will install all the development tools and additional components necessary to work with TensorFlow via the `openvino-dev` package (see **Step 4. Install the Package** on the [PyPI page](https://pypi.org/project/openvino-dev/) for parameters of other frameworks).

Then, the tools can be used by commands like:

```sh
$ mo -h
$ pot -h
```

Installation of any other dependencies is not required. For more details on the installation steps, see the [Install OpenVINO Development Tools](../../install_guides/installing-model-dev-tools.md).

## Interface Changes for Building C/C++ Applications

The new OpenVINO Runtime with its API 2.0 has also brought some changes for building C/C++ applications.

### CMake Interface

The CMake interface has been changed as follows:

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

It is possible to build applications without the CMake interface by using: MSVC IDE, UNIX makefiles, and any other interface, which has been changed as shown here:

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

OpenVINO 2022.1 introduced a reorganization of the libraries, to make deployment easier. In the previous versions, it was required to use several libraries to perform deployment steps. Now you can just use `openvino` or `openvino_c` based on your developing language,  with the necessary plugins to complete your task. For example, `openvino_intel_cpu_plugin` and `openvino_ir_frontend` plugins will enable loading OpenVINO IRs and performing inference on the CPU device (for more details, see the [Local distribution with OpenVINO](../deployment/local-distribution.md)).

Below are detailed comparisons of the library structure between OpenVINO 2022.1 and the previous versions:

* Starting with 2022.1 release, a single core library with all the functionalities (`openvino` for C++ Runtime, `openvino_c` for Inference Engine API C interface) is used, instead of the previous core libraries which contained `inference_engine`, `ngraph`, `inference_engine_transformations` and `inference_engine_lp_transformations`.
* The optional `inference_engine_preproc` preprocessing library (if `InferenceEngine::PreProcessInfo::setColorFormat` or `InferenceEngine::PreProcessInfo::setResizeAlgorithm` is used) has been renamed to `openvino_gapi_preproc` and deprecated in 2022.1. For more details, see the [Preprocessing capabilities of OpenVINO API 2.0](preprocessing.md).
* The libraries of plugins have been renamed as follows:
   * `openvino_intel_cpu_plugin` is used for [CPU](../supported_plugins/CPU.md) device instead of `MKLDNNPlugin`.
   * `openvino_intel_gpu_plugin` is used for [GPU](../supported_plugins/GPU.md) device instead of `clDNNPlugin`.
   * `openvino_auto_plugin` is used for [Auto-Device Plugin](../auto_device_selection.md).
* The plugins for reading and converting models have been changed as follows:
   * `openvino_ir_frontend` is used to read IRs instead of `inference_engine_ir_reader`.
   * `openvino_onnx_frontend` is used to read ONNX models instead of `inference_engine_onnx_reader` (with its dependencies). 
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

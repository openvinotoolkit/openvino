# Install & Deployment {#openvino_2_0_deployment}

## Introduction

"Easy to use" is one of the main points for OpenVINO 2.0 concept. This includes not only simplification of migration from frameworks to OpenVINO Toolkit, but also how OpenVINO is organized, how the development tools are used, how to develop and deploy OpenVINO-based applications.

Starting from OpenVINO 2.0, Model Optimizer, Post Training Optimization tool, Open Model Zoo tools and other Python-based Development tools are distributed via [PyPi](https://pypi.org/project/openvino-dev/) only. This simplifies user experience, because in older versions of OpenVINO development tools should be installed and used differently dependning on distribution type (OpenVINO Installer versus PyPi), but also ensures that dependencies are handled properly via `pip` package manager as well as supports virtual environments for development tools.

Regarding the OpenVINO Installer, we have tried further to reorganize package structure to have it in a good shape:

- `runtime` folder with OpenVINO Runtime include headers, libraries and CMake interfaces.
- `tools` folder with [compile_tool](../../../tools/compile_tool/README.md), [deployment manager](../../install_guides/deployment-manager-tool.md), `requirement.txt` files with link to corresponding version of `openvino-dev` package.
- `python` folder with OpenVINO Python Runtime.

## Installing development tools

In older versions of OpenVINO, Development tools were a part of main package. Once the package is installed, users need to install additional tools dependencies (e.g. `requirements_tf.txt` to convert TensorFlow models via Model Optimizer, install POT and AC tools via `setup.py` scripts), then use `setupvars` scripts to make the tools available in command line:

```sh
$ mo.py -h
```

Starting with OpenVINO 2.0 users can install development tools only from [PyPi](https://pypi.org/project/openvino-dev/) repository:

```
$ python3 -m pip install -r <openvino_root>/tools/requirements_tf.txt 
```

Which installs all OpenVINO development tools via the `openvino-dev` package and TensorFlow as an extra (see [Step 4. Install the Package](https://pypi.org/project/openvino-dev/) for details).

Then, tools can be used as:

```sh
$ mo -h
$ pot -h
```

Without installations of other dependencies. See [Install OpenVINO™ Development Tools](../../install_guides/installing-model-dev-tools.md) for more details.

## Build C / C++ applications

### CMake interface

Since OpenVINO 2.0 introduced new OpenVINO 2.0 Runtime, the CMake interface is changed as well.

Inference Engine:

```cmake
find_package(InferenceEngine REQUIRED)
find_package(ngraph REQUIRED)
add_executable(ie_ngraph_app main.cpp)
target_link_libraries(ie_ngraph_app PRIVATE ${InferenceEngine_LIBRARIES} ${NGRAPH_LIBRARIES})
```

OpenVINO Runtime 2.0:

```cmake
find_package(OpenVINO REQUIRED)
add_executable(ov_app main.cpp)
target_link_libraries(ov_app PRIVATE openvino::runtime)

add_executable(ov_c_app main.c)
target_link_libraries(ov_c_app PRIVATE openvino::runtime::c)
```

### Native interfaces

In order to build application without CMake interface, users could use MSVC IDE, Unix makefiles and any other interfaces.

Older OpenVINO packages:

@sphinxdirective

.. tab:: Include dirs

  .. code-block:: sh

    <openvino_root>/deployment_tools/inference_engine/include
    <openvino_root>/deployment_tools/ngraph/include

.. tab:: Path to libs

  .. code-block:: sh

    <openvino_root>/deployment_tools/inference_engine/lib/intel64/Release
    <openvino_root>/deployment_tools/ngraph/lib/

.. tab:: Shared libs

  .. code-block:: sh

    # Unix systems
    inference_engine.so ngraph.so

    # Windows OS
    inference_engine.dll ngraph.dll

.. tab:: (Win) .lib files

  .. code-block:: sh

    ngraph.lib
    inference_engine.lib

@endsphinxdirective

OpenVINO 2.0 package:

@sphinxdirective

.. tab:: Include dirs

  .. code-block:: sh

    <openvino_root>/runtime/include

.. tab:: Path to libs

  .. code-block:: sh

    <openvino_root>/runtime/lib/intel64/Release

.. tab:: Shared libs

  .. code-block:: sh

    # Unix systems
    openvino.so

    # Windows OS
    openvino.dll

.. tab:: (Win) .lib files

  .. code-block:: sh

    openvino.lib

@endsphinxdirective

## Deployment

### Libraries reorganization

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

So, to perform deployment steps - just take only required functionality: `openvino` or `openvino_c` depending on desired language plus plugins which are needed to solve your task. For example, `openvino_intel_cpu_plugin` and `openvino_ir_frontend` plugins to be able to load OpenVINO IRs and perform inference on CPU device.

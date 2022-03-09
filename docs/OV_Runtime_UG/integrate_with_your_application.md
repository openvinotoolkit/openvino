# Integrate OpenVINO™ with Your Application {#openvino_docs_Integrate_OV_with_your_application}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_Runtime_UG_Model_Representation
   openvino_docs_OV_Runtime_UG_Infer_request

@endsphinxdirective

> **NOTE**: Before start using OpenVINO™ Runtime, make sure you set all environment variables during the installation. If you did not, follow the instructions from the _Set the Environment Variables_ section in the installation guides:
> * [For Windows* 10](../install_guides/installing-openvino-windows.md)
> * [For Linux*](../install_guides/installing-openvino-linux.md)
> * [For macOS*](../install_guides/installing-openvino-macos.md)
> * To build an open source version, use the [OpenVINO™ Runtime Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## Use OpenVINO™ Runtime API to Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the OpenVINO™ Runtime C++ API:

![ie_api_use_cpp]

### Step 1. Create OpenVINO™ Runtime Core 

Include next files to work with OpenVINO™ Runtime:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [include]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [import]

@endsphinxdirective

Use the following code to create OpenVINO™ Core to manage available devices and read model objects:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part1]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part1]

@endsphinxdirective

### Step 2. Compile the Model

`ov::CompiledModel` class represents a device specific compiled model. `ov::CompiledModel` allows you to get information inputs or output ports by a tensor name or index.

Compile the model for a specific device using `ov::Core::compile_model()`:

@sphinxdirective

.. tab:: C++

    .. tab:: IR

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [part2_1]

    .. tab:: ONNX

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [part2_2]

    .. tab:: PaddlePaddle

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [part2_3]

    .. tab:: ov::Model

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [part2_4]

.. tab:: Python

    .. tab:: IR

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_1]

    .. tab:: ONNX

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_2]

    .. tab:: PaddlePaddle

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_3]

    .. tab:: ov::Model

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_4]

@endsphinxdirective

The `ov::Model` object represents any models inside the OpenVINO™ Runtime.
For more details please read article about [OpenVINO™ Model representation](model_representation.md).

The code above creates a compiled model associated with a single hardware device from the model object.
It is possible to create as many compiled models as needed and use them simultaneously (up to the limitation of the hardware resources).
To learn how to change the device configuration, read the [Query device properties](./supported_plugins/config_properties.md) article.

### Step 3. Create an Inference Request

`ov::InferRequest` class provides methods for model inference in OpenVINO™ Runtime. Create an infer request using the following code (see [InferRequest detailed documentation](./ov_infer_request.md) for more details):

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part3]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part3]

@endsphinxdirective

### Step 4. Set Inputs

You can use external memory to create `ov::Tensor` and use the `ov::InferRequest::set_input_tensor` method to put this tensor on the device:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part4]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part4]

@endsphinxdirective

### Step 5. Start Inference

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve application's overall frame-rate, because rather than wait for inference to complete, the app can keep working on the host, while the accelerator is busy. You can use `ov::InferRequest::start_async` to start model inference in the asynchronous mode and call `ov::InferRequest::wait` to wait for the inference results:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part5]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part5]

@endsphinxdirective

This section demonstrates a simple pipeline, to get more information about other ways to perform inference, read the dedicated ["Run inference" section](./ov_infer_request.md).

### Step 6. Process the Inference Results 

Go over the output tensors and process the inference results.

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/src/main.cpp
       :language: cpp
       :fragment: [part6]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/src/main.py
       :language: python
       :fragment: [part6]

@endsphinxdirective

## Link and Build Your C++ Application with OpenVINO™ Runtime

The example uses CMake for project configuration.

1. **Create a structure** for the project:
   ``` sh
   project/
       ├── CMakeLists.txt  - CMake file to build
       ├── ...             - Additional folders like includes/
       └── src/            - source folder
           └── main.cpp
   build/                  - build directory
       ...      
   ```

2. **Include OpenVINO™ Runtime libraries** in `project/CMakeLists.txt`

   @snippet snippets/CMakeLists.txt cmake:integration_example

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

> **NOTE**: Make sure you set environment variables first by running `<INSTALL_DIR>/setupvars.sh` (or `setupvars.bat` for Windows). Otherwise the `OpenVINO_DIR` variable won't be configured properly to pass `find_package` calls.

```sh
cd build/
cmake ../project
cmake --build .
```
It's allowed to specify additional build options (e.g. to build CMake project on Windows with a specific build tools). Please refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)) for details.

## Run Your Application

Congratulations, you have made your first application with OpenVINO™ toolkit, now you may run it.

## See also

 - [OpenVINO™ Runtime Preprocessing](./preprocessing_overview.md)

[ie_api_flow_cpp]: img/BASIC_IE_API_workflow_Cpp.svg
[ie_api_use_cpp]: img/IMPLEMENT_PIPELINE_with_API_C.svg
[ie_api_flow_python]: img/BASIC_IE_API_workflow_Python.svg
[ie_api_use_python]: img/IMPLEMENT_PIPELINE_with_API_Python.svg

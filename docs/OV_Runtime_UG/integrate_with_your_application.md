# Integrate OpenVINO™ with Your Application {#openvino_docs_OV_UG_Integrate_OV_with_your_application}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Model_Representation
   openvino_docs_OV_UG_Infer_request
   openvino_docs_OV_UG_Python_API_exclusives

@endsphinxdirective

> **NOTE**: Before start using OpenVINO™ Runtime, make sure you set all environment variables during the installation. If you did not, follow the instructions from the _Set the Environment Variables_ section in the installation guides:
> * [For Windows 10](../install_guides/installing-openvino-windows.md)
> * [For Linux](../install_guides/installing-openvino-linux.md)
> * [For macOS](../install_guides/installing-openvino-macos.md)
> * To build an open source version, use the [OpenVINO™ Runtime Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## Using OpenVINO™ Runtime API to Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the OpenVINO™ Runtime C++ or Python API:

![ie_api_use_cpp]

### Step 1. Create OpenVINO™ Runtime Core 

Include next files to work with OpenVINO™ Runtime:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp include

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py import

@endsphinxtab

@endsphinxtabset

Use the following code to create OpenVINO™ Core to manage available devices and read model objects:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part1

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part1

@endsphinxtab

@endsphinxtabset

### Step 2. Compile the Model

`ov::CompiledModel` class represents a device specific compiled model. `ov::CompiledModel` allows you to get information inputs or output ports by a tensor name or index. This approach is aligned with the majority of frameworks.

Compile the model for a specific device using `ov::Core::compile_model()`:

@sphinxtabset

@sphinxtab{C++}

@sphinxtabset

@sphinxtab{IR}

@snippet docs/snippets/src/main.cpp part2_1

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/src/main.cpp part2_2

@endsphinxtab

@sphinxtab{PaddlePaddle}

@snippet docs/snippets/src/main.cpp part2_3

@endsphinxtab

@sphinxtab{ov::Model}

@snippet docs/snippets/src/main.cpp part2_4

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Python}

@sphinxtabset

@sphinxtab{IR}

@snippet docs/snippets/src/main.py part2_1

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/snippets/src/main.py part2_2

@endsphinxtab

@sphinxtab{PaddlePaddle}

@snippet docs/snippets/src/main.py part2_3

@endsphinxtab

@sphinxtab{ov::Model}

@snippet docs/snippets/src/main.py part2_4

@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

The `ov::Model` object represents any models inside the OpenVINO™ Runtime.
For more details, refer to the [OpenVINO™ Model representation](model_representation.md) article.

The code above creates a compiled model associated with a single hardware device from the model object.
It is possible to create as many compiled models as needed and use them simultaneously (up to the limitation of the hardware resources).
To learn how to change the device configuration, read the [Query device properties](./supported_plugins/config_properties.md) article.

### Step 3. Create an Inference Request

The `ov::InferRequest` class provides methods for model inference in OpenVINO™ Runtime. For more details see the [InferRequest detailed documentation](./ov_infer_request.md). 

Use the following code to create an inference request:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part3

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part3

@endsphinxtab

@endsphinxtabset

### Step 4. Set Inputs

You can use external memory to create `ov::Tensor` and use the `ov::InferRequest::set_input_tensor` method to put this tensor on the device:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part4

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part4

@endsphinxtab

@endsphinxtabset

### Step 5. Start Inference

OpenVINO™ Runtime supports inference in either synchronous or asynchronous mode. Using the Async API can improve overall application frame-rate, because rather than wait for inference to complete, the app can keep working on the host, while the accelerator is busy. You can use `ov::InferRequest::start_async` to start model inference in the asynchronous mode and call `ov::InferRequest::wait` to wait for the inference results:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part5

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part5

@endsphinxtab

@endsphinxtabset

This section demonstrates a simple pipeline. To get more information about other ways to perform inference, read the dedicated ["Run inference" section](./ov_infer_request.md).

### Step 6. Process the Inference Results 

Go over the output tensors and process the inference results.

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/src/main.cpp part6

@endsphinxtab

@sphinxtab{Python}

@snippet docs/snippets/src/main.py part6

@endsphinxtab

@endsphinxtabset

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

> **NOTE**: Make sure you set environment variables first by running `<INSTALL_DIR>/setupvars.sh` (or `setupvars.bat` for Windows). Otherwise the `OpenVINO_DIR` variable will not be configured properly to pass `find_package` calls.

```sh
cd build/
cmake ../project
cmake --build .
```

It is allowed to specify additional build options (e.g. to build CMake project on Windows with a specific build tools). For details, refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)).


## Additional Resources

 - See the [OpenVINO Samples](Samples_Overview.md) page or the [Open Model Zoo Demos](https://docs.openvino.ai/latest/omz_demos.html) page for specific examples of how OpenVINO pipelines are implemented for applications like image classification, text prediction, and many others.
 - [OpenVINO™ Runtime Preprocessing](./preprocessing_overview.md)
 - [Using Encrypted Models with OpenVINO&trade;](./protecting_model_guide.md)
 - [OpenVINO Samples](Samples_Overview.md)
 - [Open Model Zoo Demos](https://docs.openvino.ai/latest/omz_demos.html)

[ie_api_flow_cpp]: img/BASIC_IE_API_workflow_Cpp.svg
[ie_api_use_cpp]: img/IMPLEMENT_PIPELINE_with_API_C.svg
[ie_api_flow_python]: img/BASIC_IE_API_workflow_Python.svg
[ie_api_use_python]: img/IMPLEMENT_PIPELINE_with_API_Python.svg

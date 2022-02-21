# Integrate OpenVINO™ into your application {#openvino_docs_Integrate_OV_into_customer_application}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_Runtime_UG_Model_Representation
   openvino_docs_OV_Runtime_UG_Infer_request

@endsphinxdirective

The following diagram illustrates usual application development workflow:

![ie_api_flow_cpp]

Read the sections below to learn about each item.

> **NOTE**: Before start using OpenVINO™ Runtime, make sure you set all environment variables during the installation. If you did not, follow the instructions from the _Set the Environment Variables_ section in the installation guides:
> * [For Windows* 10](../install_guides/installing-openvino-windows.md)
> * [For Linux*](../install_guides/installing-openvino-linux.md)
> * [For macOS*](../install_guides/installing-openvino-macos.md)
> * To build an open source version, use the [OpenVINO™ Runtime Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## OpenVINO™ Model Representation

Before the start, it is necessary to say several words about OpenVINO™ Model representation.
In OpenVINO™ Runtime a model is represented by the `ov::Model` class.

The `ov::Model` object represents any models inside the OpenVINO™ Runtime.
`ov::Model` allows to use tensor names or indexes to work wit model inputs/outpus. To get model input/output ports you can use `ov::Model::inputs()` or `ov::Model::outputs()` respectively.
For more details please read article about [OpenVINO™ Model representation](model_representation.md).

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

`ov::CompiledModel` class represents device specific compiled model. `ov::CompiledModel` allows to get information inputs or output ports by tensor name or index.

Compile the model to the device using `ov::Core::compile_model()`:

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

    .. tab:: Paddle

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [part2_3]

    .. tab:: OpenVINO™ Model

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

    .. tab:: Paddle

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_3]

    .. tab:: OpenVINO™ Model

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [part2_4]

@endsphinxdirective

It creates a compiled model from a model object. The compiled model is associated with single hardware device.
It is possible to create as many compiled models as needed and to use them simultaneously (up to the limitation of the hardware resources).
Please read article about [OpenVINO™ Device Properties API](InferenceEngine_QueryAPI.md) to understand how device configuration can be changed.

### Step 3. Create an Inference Request

`ov::InferRequest` class provides methods for inference model inside the OpenVINO™ runtime.
Create an infer request using the following code:

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

You can use external memory to create `ov::Tensor` and use `ov::InferRequest::set_input_tensor()` method to put this tensor on device:

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

OpenVINO™ Runtime supports inference in asynchronous or synchronous mode. Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete, the app can continue doing things on the host, while accelerator is busy. You can use `ov::InferRequest::start_async()` to infer model in asynchronous mode and call `ov::InferRequest::wait()` for waiting inference results:

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

Asynchronous mode supports two ways to wait inference results:
  * `ov::InferRequest::wait_for()` - specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
  * `ov::InferRequest::wait()` - waits until inference result becomes available

Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Multiple requests for single `CompiledModel` are executed sequentially one by one in FIFO order.

While request is ongoing, all its methods except `ov::InferRequest::wait` or `ov::InferRequest::wait_for` would throw
the ov::Busy exception that request is busy with computations.

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

## Link with OpenVINO™ Runtime (for C++)

The example uses CMake for project configuration.

1. **Create a structure** for the project:
   ``` sh
   project/
       ├── CMakeLists.txt  - CMake file to build
       ├── ...             - Additional folders like includes/
       └── src/            - source folder
           ├── main.cpp
           └── main.py
   build/                  - build directory
       ...      
   ```

2. **Include OpenVINO™ Runtime libraries** in `project/CMakeLists.txt`

   @snippet snippets/CMakeLists.txt cmake:integration_example


## Build Your C++ Application

For details about building your application, refer to the CMake files for the sample applications.
All samples source code is located in the `<INSTALL_DIR>/samples` directory, where `INSTALL_DIR` is the OpenVINO™ installation directory.

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

> **NOTE**: Make sure you set environment variables first by running `<INSTALL_DIR>/setupvars.sh` (or `setupvars.bat` for Windows). Otherwise the `OpenVINO_DIR` variable won't be configured properly to pass `find_package` calls.

```sh
cd build/
cmake ../project
cmake --build .
```
It's allowed to specify additional build options (e.g. to build CMake project on Windows with a specific build tools). Please refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)) for details.

## Run Your Application

> **NOTE**: Before running, make sure you completed **Set the Environment Variables** section in [OpenVINO Installation](../../samples/cpp/hello_nv12_input_classification/README.md) document so that the application can find the libraries.

To run compiled applications on Microsoft* Windows* OS, make sure that Microsoft* Visual C++ 2019 is installed and
`<INSTALL_DIR>/bin/intel64/Release/*.dll` files are placed to the
application folder or accessible via `%PATH%` environment variable.

## FAQ

 - How can I use tensor name to get or set tensor?

    @sphinxdirective
    .. raw:: html

        <div class="collapsible-section">
    @endsphinxdirective

    To use tensor name in order to get or set tensor you can use `ov::InferRequest::get_tensor()` or `ov::InferRequest::set_tensor()` methods respectively:

    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [faq:get_set_tensor]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [faq:get_set_tensor]

    @endsphinxdirective

    @sphinxdirective
    .. raw:: html

        </div>
    @endsphinxdirective

 - Can I use use output tensor of one model as input for other model?
    @sphinxdirective
    .. raw:: html

        <div class="collapsible-section">
    @endsphinxdirective

    Get output tensor from the first request using `ov::InferRequest::get_tensor()` and set it as input for the second request using `ov::InferRequest::set_tensor()`. But be careful, shared tensors across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.

    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [faq:cascade_models]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [faq:cascade_models]

    @endsphinxdirective

    @sphinxdirective
    .. raw:: html

        </div>
    @endsphinxdirective

 - Can I create ROI tensor?

    @sphinxdirective
    .. raw:: html

        <div class="collapsible-section">
    @endsphinxdirective

    It is possible to re-use shared input by several models. You do not need to allocate separate input tensor for a model if it processes a ROI object located inside of already allocated input of a previous model. For instance, when first model detects objects on a video frame (stored as input tensor) and second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input tensor (used by first model) by second model and just crop ROI without allocation of new memory using `ov::Tensor()` with passing of `ov::Tensor` and `ov::Coordinate` as parameters.


    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [faq:roi_tensor]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [faq:roi_tensor]

    @endsphinxdirective

    @sphinxdirective
    .. raw:: html

        </div>
    @endsphinxdirective

 - How can I run inference in the synchronous mode?

    @sphinxdirective
    .. raw:: html

        <div class="collapsible-section">
    @endsphinxdirective

    Run inference in the synchronous mode:

    @sphinxdirective

    .. tab:: C++

        .. doxygensnippet:: docs/snippets/src/main.cpp
           :language: cpp
           :fragment: [faq:sync_infer]

    .. tab:: Python

        .. doxygensnippet:: docs/snippets/src/main.py
           :language: python
           :fragment: [faq:sync_infer]

    @endsphinxdirective

    @sphinxdirective
    .. raw:: html

        </div>
    @endsphinxdirective

## See also

 - [OpenVINO™ Runtime Preprocessing API]()

[ie_api_flow_cpp]: img/BASIC_IE_API_workflow_Cpp.svg
[ie_api_use_cpp]: img/IMPLEMENT_PIPELINE_with_API_C.svg
[ie_api_flow_python]: img/BASIC_IE_API_workflow_Python.svg
[ie_api_use_python]: img/IMPLEMENT_PIPELINE_with_API_Python.svg

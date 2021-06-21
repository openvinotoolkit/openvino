Integrate the Inference Engine with Your Application {#openvino_docs_IE_DG_Integrate_with_customer_application_new_API}
===============================

This section provides a high-level description of the process of integrating the Inference Engine into your application.
Refer to the [Hello Classification Sample](../../inference-engine/samples/hello_classification/README.md) sources
for example of using the Inference Engine in applications.

## Use the Inference Engine API in Your Code

The core `libinference_engine.so` library implements loading and parsing a model Intermediate Representation (IR), and triggers inference using a specified device. The core library has the following API:

* `InferenceEngine::Core`
* `InferenceEngine::Blob`, `InferenceEngine::TBlob`,
  `InferenceEngine::NV12Blob`
* `InferenceEngine::BlobMap`
* `InferenceEngine::InputsDataMap`, `InferenceEngine::InputInfo`,
* `InferenceEngine::OutputsDataMap`

C++ Inference Engine API wraps the capabilities of core library:

* `InferenceEngine::CNNNetwork`
* `InferenceEngine::ExecutableNetwork`
* `InferenceEngine::InferRequest`

## Integration Steps

Integration process includes the following steps:
![integration_process]

1) **Create Inference Engine Core** to manage available devices and read network objects:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part0

2) **Read a model IR** created by the Model Optimizer (.xml is supported format):

@snippet snippets/Integrate_with_customer_application_new_API.cpp part1

**Or read the model from ONNX format** (.onnx and .prototxt are supported formats). You can find more information about the ONNX format support in the document [ONNX format support in the OpenVINO™](./ONNX_Support.md).

@snippet snippets/Integrate_with_customer_application_new_API.cpp part2

3) **Configure input and output**. Request input and output information using `InferenceEngine::CNNNetwork::getInputsInfo()`, and `InferenceEngine::CNNNetwork::getOutputsInfo()`
methods:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part3

  Optionally, set the number format (precision) and memory layout for inputs and outputs. Refer to the
  [Supported configurations](supported_plugins/Supported_Devices.md) chapter to choose the relevant configuration.

  You can also allow input of any size. To do this, mark each input as resizable by setting a desired resize algorithm (e.g. `BILINEAR`) inside of the appropriate input info.

  Basic color format conversions are supported as well. By default, the Inference Engine assumes
  that the input color format is `BGR` and color format conversions are disabled. The Inference
  Engine supports the following color format conversions:
  * `RGB->BGR`
  * `RGBX->BGR`
  * `BGRX->BGR`
  * `NV12->BGR`

  where `X` is a channel that will be ignored during inference. To enable the conversions, set a
  desired color format (for example, `RGB`) for each input inside of the appropriate input info.

  If you want to run inference for multiple images at once, you can use the built-in batch
  pre-processing functionality.

> **NOTE**: Batch pre-processing is not supported if input color format is set to `ColorFormat::NV12`.

  You can use the following code snippet to configure input and output:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part4

> **NOTE**: NV12 input color format pre-processing differs from other color conversions. In case of NV12,
>  Inference Engine expects two separate image planes (Y and UV). You must use a specific
>  `InferenceEngine::NV12Blob` object instead of default blob object and set this blob to
>  the Inference Engine Infer Request using `InferenceEngine::InferRequest::SetBlob()`.
>  Refer to [Hello NV12 Input Classification C++ Sample](../../inference-engine/samples/hello_nv12_input_classification/README.md)
>  for more details.

  If you skip this step, the default values are set:

  * no resize algorithm is set for inputs
  * input color format - `ColorFormat::RAW` meaning that input does not need color
    conversions
  * input and output precision - `Precision::FP32`
  * input layout - `Layout::NCHW`
  * output layout depends on number of its dimensions:

|Number of dimensions |  5    |  4   |   3 |  2 |  1 |
|:--------------------|-------|------|-----|----|----|
|Layout               | NCDHW | NCHW | CHW | NC | C  |

4) **Load the model** to the device using `InferenceEngine::Core::LoadNetwork()`:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part5

    It creates an executable network from a network object. The executable network is associated with single hardware device.
    It is possible to create as many networks as needed and to use them simultaneously (up to the limitation of the hardware resources).
    Third parameter is a configuration for plugin. It is map of pairs: (parameter name, parameter value). Choose device from
     [Supported devices](supported_plugins/Supported_Devices.md) page for more details about supported configuration parameters.

@snippet snippets/Integrate_with_customer_application_new_API.cpp part6

5) **Create an infer request**:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part7

6) **Prepare input**. You can use one of the following options to prepare input:
    * **Optimal way for a single network.** Get blobs allocated by an infer request using `InferenceEngine::InferRequest::GetBlob()` and feed an image and the input data to the blobs. In this case, input data must be aligned (resized manually) with a given blob size and have a correct color format.

@snippet snippets/Integrate_with_customer_application_new_API.cpp part8

    * **Optimal way for a cascade of networks (output of one network is input for another).** Get output blob from the first request using `InferenceEngine::InferRequest::GetBlob()` and set it as input for the second request using `InferenceEngine::InferRequest::SetBlob()`.

@snippet snippets/Integrate_with_customer_application_new_API.cpp part9

    * **Optimal way to handle ROI (a ROI object located inside of input of one network is input for another).** It is possible to re-use shared input by several networks. You do not need to allocate separate input blob for a network if it processes a ROI object located inside of already allocated input of a previous network. For instance, when first network detects objects on a video frame (stored as input blob) and second network accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input blob (used by first network) by second network and just crop ROI without allocation of new memory using `InferenceEngine::make_shared_blob()` with passing of `InferenceEngine::Blob::Ptr` and `InferenceEngine::ROI` as parameters.

@snippet snippets/Integrate_with_customer_application_new_API.cpp part10

Make sure that shared input is kept valid during execution of each network. Otherwise, ROI blob may be corrupted if the original input blob (that ROI is cropped from) has already been rewritten.

    * Allocate input blobs of the appropriate types and sizes, feed an image and the input data to the blobs, and call `InferenceEngine::InferRequest::SetBlob()` to set these blobs for an infer request:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part11

      A blob can be filled before and after `SetBlob()`.

> **NOTE:**
>
> * The `SetBlob()` method compares precision and layout of an input blob with the ones defined in step 3 and
> throws an exception if they do not match. It also compares a size of the input blob with input
> size of the read network. But if input was configured as resizable, you can set an input blob of
> any size (for example, any ROI blob). Input resize will be invoked automatically using resize
> algorithm configured on step 3. Similarly to the resize, color format conversions allow the color
> format of an input blob to differ from the color format of the read network. Color format
> conversion will be invoked automatically using color format configured on step 3.
>
> * `GetBlob()` logic is the same for pre-processable and not pre-processable input. Even if it is
> called with input configured as resizable or as having specific color format, a blob allocated by
> an infer request is returned. Its size and color format are already consistent with the
> corresponding values of the read network. No pre-processing will happen for this blob. If you
> call `GetBlob()` after `SetBlob()`, you will get the blob you set in `SetBlob()`.

7) **Do inference** by calling the `InferenceEngine::InferRequest::StartAsync` and `InferenceEngine::InferRequest::Wait` methods for asynchronous request:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part12

or by calling the `InferenceEngine::InferRequest::Infer` method for synchronous request:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part13

`StartAsync` returns immediately and starts inference without blocking main thread, `Infer` blocks
 main thread and returns when inference is completed. Call `Wait` for waiting result to become available for asynchronous request.

There are three ways to use it:
* specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
* `InferenceEngine::InferRequest::WaitMode::RESULT_READY` - waits until inference result becomes available
* `InferenceEngine::InferRequest::WaitMode::STATUS_ONLY` - immediately returns request status.It does not
block or interrupts current thread.

Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Multiple requests for single `ExecutableNetwork` are executed sequentially one by one in FIFO order.

While request is ongoing, all its methods except `InferenceEngine::InferRequest::Wait` would throw an
exception.

8) Go over the output blobs and **process the results**.
Note that casting `Blob` to `TBlob` via `std::dynamic_pointer_cast` is not the recommended way. It's better to access data via the `buffer()` and `as()` methods as follows:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part14

## Build Your Application

For details about building your application, refer to the CMake files for the sample applications.
All samples source code is located in the `<INSTALL_DIR>/openvino/inference_engine/samples` directory, where `INSTALL_DIR` is the OpenVINO™ installation directory.

### CMake project creation

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

2. **Include Inference Engine, nGraph and OpenCV libraries** in `project/CMakeLists.txt`  
[OpenCV](https://docs.opencv.org/master/db/df5/tutorial_linux_gcc_cmake.html) integration is needed mostly for pre-processing input data and ngraph for more complex applications using [ngraph API](../nGraph_DG/nGraph_dg.md).
``` cmake
cmake_minimum_required(VERSION 3.0.0)
project(project_name)
find_package(ngraph REQUIRED)
find_package(InferenceEngine REQUIRED)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} ${NGRAPH_LIBRARIES})
```
3. **To build your project** using CMake with the default build tools currently available on your machine, execute the following commands:
> **NOTE**: Make sure you set environment variables first by running `<INSTALL_DIR>/bin/setupvars.sh` (or setupvars.bat for Windows)`. Otherwise the `InferenceEngine_DIR` and `OpenCV_DIR` variables won't be configured properly to pass `find_package` calls.
```sh
cd build/
cmake ../project
cmake --build .
```
It's allowed to specify additional build options (e.g. to build CMake project on Windows with a specific build tools). Please refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)) for details.

### Run Your Application

> **NOTE**: Before running, make sure you completed **Set the Environment Variables** section in [OpenVINO Installation](../../inference-engine/samples/hello_nv12_input_classification/README.md) document so that the application can find the libraries.

To run compiled applications on Microsoft* Windows* OS, make sure that Microsoft* Visual C++ 2017
Redistributable and Intel® C++ Compiler 2017 Redistributable packages are installed and
`<INSTALL_DIR>/bin/intel64/Release/*.dll` files are placed to the
application folder or accessible via `%PATH%` environment variable.

[integration_process]: img/integration_process.png

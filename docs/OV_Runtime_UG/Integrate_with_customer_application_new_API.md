# Integrate Inference Engine {#openvino_docs_IE_DG_Integrate_with_customer_application_new_API}

## Integrate Inference Engine with Your C++ Application

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The following diagram illustrates the typical Inference Engine С++ API workflow:

![ie_api_flow_cpp]

Read the sections below to learn about each item.

> **NOTE**: Before start using Inference Engine, make sure you set all environment variables during the installation. If you did not, follow the instructions from the _Set the Environment Variables_ section in the installation guides:
> * [For Windows* 10](../install_guides/installing-openvino-windows.md)
> * [For Linux*](../install_guides/installing-openvino-linux.md)
> * [For macOS*](../install_guides/installing-openvino-macos.md)
> * To build an open source version, use the [Inference Engine Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

### Link with Inference Library

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
[OpenCV](https://docs.opencv.org/master/db/df5/tutorial_linux_gcc_cmake.html) integration is needed mostly for pre-processing input data and model representation in OpenVINO™ Runtime for more complex applications using [OpenVINO Model API](../OV_Runtime_UG/model_representation.md).
   ``` cmake
   cmake_minimum_required(VERSION 3.0.0)
   project(project_name)
   find_package(OpenVINO REQUIRED)
   add_executable(${PROJECT_NAME} src/main.cpp)
   target_link_libraries(${PROJECT_NAME} PRIVATE openvino::runtime)
   ```

### Use Inference Engine API to Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the Inference Engine C++ API:   

![ie_api_use_cpp]
#### Step 1. Create Inference Engine Core 

Use the following code to create Inference Engine Core to manage available devices and read network objects:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part0

#### Step 2 (Optional). Configure Input and Output of the Model

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective
    

Optionally, configure input and output of the model using the steps below:

1. Load a model to a Core object:
   @sphinxdirective
   
   .. tab:: IR
   
      .. code-block:: c
   
         auto network  = core.ReadNetwork("model.xml");

   .. tab:: ONNX
      
      .. code-block:: c
         
         auto network = core.ReadNetwork("model.onnx");

      You can find more information about the ONNX format support in the document `ONNX format support in the OpenVINO™ <https://docs.openvino.ai/latest/openvino_docs_IE_DG_ONNX_Support.html>`_   
   
   .. tab:: nGraph
      
      .. code-block:: c
         
         std::shared_ptr<Function> createNetwork() {
            // To construct a network, please follow 
            // https://docs.openvino.ai/latest/openvino_docs_nGraph_DG_build_function.html
         }
         auto network = CNNNetwork(createNetwork());

   @endsphinxdirective

2. Request input and output information using `InferenceEngine::CNNNetwork::getInputsInfo()`, and `InferenceEngine::CNNNetwork::getOutputsInfo()` methods:
   ```cpp
   /** Take information about all topology inputs **/
   InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
   /** Iterate over all input info**/
   for (auto &item : input_info) {
       auto input_data = item.second;
           // Add your input configuration steps here
   }
   
   /** Take information about all topology outputs **/
   InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();
   /** Iterate over all output info**/
   for (auto &item : output_info) {
       auto output_data = item.second;
           // Add your output configuration steps here
   }
   ```
   Configuring options:
   1. **Set precision** (number format): FP16, FP32, INT8, etc. Refer to the Supported Configurations section on the [Supported Devices](supported_plugins/Supported_Devices.md) page to choose the relevant configuration.<br>
   For input (*iterate over all input info*):
   ```cpp
   input_data->setPrecision(InferenceEngine::Precision::U8);
   ```
   For output  (*iterate over all output info*):
   ```cpp
   output_data->setPrecision(InferenceEngine::Precision::FP32);
   ```
   **By default**, the input and output precision is set to `Precision::FP32`.

   2. **Set layout** (NCHW, ).<br>
   For input (*iterate over all input info*):
   ```cpp
   input_data->setLayout(InferenceEngine::Layout::NCHW);
   ```
   **By default**, the input layout is set to `Layout::NCHW`.<br>
   For output (*iterate over all output info*):
   ```cpp
   output_data->setLayout(InferenceEngine::Layout::NC);
   ```
      **By default**, the output layout depends on a number of its dimensions:<br>
      |Number of dimensions |  5    |  4   |   3 |  2 |  1 |
      |:--------------------|-------|------|-----|----|----|
      |Layout               | NCDHW | NCHW | CHW | NC | C  |
   3. **Set resize algorithm for inputs** (Bilinear). You can allow input of any size. To do this, mark each input as resizable by setting a desired resize algorithm (e.g. `BILINEAR`) inside of the appropriate input info (*Iterate over all input info*):
   ```cpp
   input_data->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
   ```
   **By default**, no resize algorithm is set for inputs.

   4. **Set color format** (BGR, RGB, NV12). Basic color format conversions are supported as well. **By default**, the Inference Engine assumes that the input color format is BGR and color format conversions are disabled. Set `ColorFormat::RAW` input color format if the input does not need color conversions. The Inference Engine supports the following color format conversions:
      * RGB->BGR
      * RGBX->BGR
      * BGRX->BGR
      * NV12->BGR
      where X is a channel that will be ignored during inference. To enable the conversions, set a desired color format (for example, RGB) for each input inside of the appropriate input info (*iterate over all input info*):
   ```cpp
   input_data->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);
   ```
   > **NOTE**: NV12 input color format pre-processing differs from other color conversions. In case of NV12, Inference Engine expects two separate image planes (Y and UV). You must use a specific `InferenceEngine::NV12Blob` object instead of default blob object and set this blob to the Inference Engine Infer Request using `InferenceEngine::InferRequest::SetBlob()`. Refer to [Hello NV12 Input Classification C++ Sample](../../samples/cpp/hello_nv12_input_classification/README.md) for more details.
   
   5. **Run on multiple images** with setting batch. If you want to run inference for multiple images at once, you can use the built-in batch pre-processing functionality.
   
      **NOTE** : Batch pre-processing is not supported if input color format is set to `ColorFormat::NV12`.

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective

#### Step 3. Load the Model to the Device

Load the model to the device using `InferenceEngine::Core::LoadNetwork()`:


@sphinxdirective
   
.. tab:: IR

   .. code-block:: c

      executable_network = core.LoadNetwork("model.xml", "CPU");

.. tab:: ONNX

   .. code-block:: c

      executable_network = core.LoadNetwork("model.onnx", "CPU");

.. tab:: nGraph

   .. code-block:: c

      std::shared_ptr<Function> createNetwork() {
         // To construct a network, please follow 
         // https://docs.openvino.ai/latest/openvino_docs_nGraph_DG_build_function.html
      }
      auto network = CNNNetwork(createNetwork());
      executable_network = core.LoadNetwork(network, "CPU");

.. tab:: Model From Step 2
   
   Follow this step only if you went through optional "Step 2 (Optional). Configure Input and Output of the Model", otherwise use another tab for your model type: IR (OpenVINO Intermediate Representation), ONNX or nGraph.
   
   .. code-block:: c

      executable_network = core.LoadNetwork(network, "CPU");

@endsphinxdirective


It creates an executable network from a network object. The executable network is associated with single hardware device.
It is possible to create as many networks as needed and to use them simultaneously (up to the limitation of the hardware resources).

Third parameter is a configuration for plugin. It is map of pairs: (parameter name, parameter value). Choose device from
[Supported devices](supported_plugins/Supported_Devices.md) page for more details about supported configuration parameters.

@snippet snippets/Integrate_with_customer_application_new_API.cpp part6

#### Step 4. Create an Inference Request

Create an infer request using the following code:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part7

#### Step 5. Prepare Input 

You can use one of the following options to prepare input:

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

> **NOTE**:
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

#### Step 6. Start Inference

Start inference in asynchronous or synchronous mode. Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete, the app can continue doing things on the host, while accelerator is busy.

* For synchronous inference request:
   ```cpp
   infer_request.Infer();
   ```

* For asynchronous inference request: 
  ```cpp
  infer_request.StartAsync();
  infer_request.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
  ```
  `StartAsync` returns immediately and starts inference without blocking main thread, `Infer` blocks main thread and returns when inference is completed. Call `Wait` for waiting result to become available for asynchronous request.

  There are three ways to use it:
      * specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
      * `InferenceEngine::InferRequest::WaitMode::RESULT_READY` - waits until inference result becomes available
      * `InferenceEngine::InferRequest::WaitMode::STATUS_ONLY` - immediately returns request status.It does not
      block or interrupts current thread.
   

Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Multiple requests for single `ExecutableNetwork` are executed sequentially one by one in FIFO order.

While request is ongoing, all its methods except `InferenceEngine::InferRequest::Wait` would throw an
exception.

#### Step 7. Process the Inference Results 

Go over the output blobs and process the inference results. Note that casting `Blob` to `TBlob` via `std::dynamic_pointer_cast` is not the recommended way. It's better to access data via the `buffer()` and `as()` methods as follows:

@snippet snippets/Integrate_with_customer_application_new_API.cpp part14

### Build Your Application

For details about building your application, refer to the CMake files for the sample applications.
All samples source code is located in the `<INSTALL_DIR>/samples` directory, where `INSTALL_DIR` is the OpenVINO™ installation directory.

To build your project using CMake with the default build tools currently available on your machine, execute the following commands:

> **NOTE**: Make sure you set environment variables first by running `<INSTALL_DIR>/setupvars.sh` (or `setupvars.bat` for Windows). Otherwise the `InferenceEngine_DIR` and `OpenCV_DIR` variables won't be configured properly to pass `find_package` calls.

```sh
cd build/
cmake ../project
cmake --build .
```
It's allowed to specify additional build options (e.g. to build CMake project on Windows with a specific build tools). Please refer to the [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)) for details.

### Run Your Application

> **NOTE**: Before running, make sure you completed **Set the Environment Variables** section in [OpenVINO Installation](../../samples/cpp/hello_nv12_input_classification/README.md) document so that the application can find the libraries.

To run compiled applications on Microsoft* Windows* OS, make sure that Microsoft* Visual C++ 2017
Redistributable and Intel® C++ Compiler 2017 Redistributable packages are installed and
`<INSTALL_DIR>/bin/intel64/Release/*.dll` files are placed to the
application folder or accessible via `%PATH%` environment variable.

## Integrate Inference Engine with Your Python Application

@sphinxdirective
.. raw:: html

    <div id="switcher-python" class="switcher-anchor">Python</div>
@endsphinxdirective

This document explains how to integrate and use the Inference Engine API with your Python application.   

The following diagram illustrates the typical Inference Engine Python API workflow:
![ie_api_flow_python] 

Read the sections below to learn about each item.

### Import Inference Module

To make use of the Inference Engine functionality, import IECore to your application: 

```py
from openvino.inference_engine import IECore
``` 
 
### Use Inference Engine API 

This section provides step-by-step instructions to implement a typical inference pipeline with the Inference Engine API:   

![ie_api_use_python]

#### Step 1. Create Inference Engine Core

Use the following code to create Inference Engine Core to manage available devices and read network objects: 
```py
ie = IECore()
``` 
#### Step 2 (Optional). Read model. Configure Input and Output of the Model

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective

Optionally, configure input and output of the model using the steps below: 

1. Read model 
   @sphinxdirective
      
   .. tab:: IR
   
      .. code-block:: python
   
         net = ie.read_network(model="model.xml")
   
   .. tab:: ONNX
      
      .. code-block:: python
         
         net = ie.read_network(model="model.onnx")
   
   .. tab:: nGraph
      
      .. code-block:: python
         
         #Basic example of nGraph model creation
         param = Parameter(Type.f32, Shape([1, 3, 22, 22]))
         relu = ng.relu(param)
         func = Function([relu], [param], 'test')
         caps = Function.to_capsule(func)
         net = IENetwork(caps)
   
   @endsphinxdirective

2. Request input and output information using input_info, outputs 
   ```py
   inputs = net.input_info 
   input_name = next(iter(net.input_info))  

   outputs = net.outputs 
   output_name = next(iter(net.outputs)) 
   ``` 
   Information for this input layer is stored in input_info. The next cell prints the input layout, precision and shape. 
   ```py
   print("Inputs:")
   for name, info in net.input_info.items():
       print("\tname: {}".format(name))
       print("\tshape: {}".format(info.tensor_desc.dims))
       print("\tlayout: {}".format(info.layout))
       print("\tprecision: {}\n".format(info.precision))
   ```
   This cell output tells us that the model expects inputs with a shape of [1,3,224,224], and that this is in NCHW layout. This means that the model expects input data with a batch size (N) of 1, 3 channels (C), and images of a height (H) and width (W) of 224. The input data is expected to be of FP32 (floating point) precision. 
    
   Getting the output layout, precision and shape is similar to getting the input layout, precision and shape. 
   ```py
   print("Outputs:")
   for name, info in net.outputs.items():
       print("\tname: {}".format(name))
       print("\tshape: {}".format(info.shape))
       print("\tlayout: {}".format(info.layout))
       print("\tprecision: {}\n".format(info.precision))
   ```
   This cell output shows that the model returns outputs with a shape of [1, 1001], where 1 is the batch size (N) and 1001 the number of classes (C). The output is returned as 32-bit floating point. 

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective 

#### Step 3. Load model to the Device 

Load the model to the device using `load_network()`:

@sphinxdirective
   
.. tab:: IR

   .. code-block:: python

      exec_net = ie.load_network(network= "model.xml", device_name="CPU") 
.. tab:: ONNX
   
   .. code-block:: python
      
      exec_net = ie.load_network(network= "model.onnx", device_name="CPU") 

.. tab:: Model from step 2
   
   .. code-block:: python
   
      exec_net = ie.load_network(network=net, device_name="CPU")

@endsphinxdirective

This example is designed for CPU device, refer to the [Supported Devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) page to read about more devices. 

#### Step 4. Prepare input 
```py
import cv2 
import numpy as np 

image = cv2.imread("image.png") 

# Resize with OpenCV your image if needed to match with net input shape 
# N, C, H, W = net.input_info[input_name].tensor_desc.dims
# image = cv2.resize(src=image, dsize=(W, H)) 

# Converting image to NCHW format with FP32 type 
input_data = np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32) 
```

#### Step 5. Start Inference
```py
result = exec_net.infer({input_name: input_data}) 
``` 

#### Step 6. Process the Inference Results 
```py
output = result[output_name] 
```

### Run Your Application

Congratulations, you have made your first Python application with OpenVINO™ toolkit, now you may run it.

[ie_api_flow_cpp]: img/BASIC_IE_API_workflow_Cpp.svg
[ie_api_use_cpp]: img/IMPLEMENT_PIPELINE_with_API_C.svg
[ie_api_flow_python]: img/BASIC_IE_API_workflow_Python.svg
[ie_api_use_python]: img/IMPLEMENT_PIPELINE_with_API_Python.svg

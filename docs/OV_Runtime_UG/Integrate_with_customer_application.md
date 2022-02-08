# Integrate OpenVINO™ into customer application {#openvino_docs_Integrate_OV_into_customer_application}

## Integrate OpenVINO™ Runtime with Your C++ Application

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The following diagram illustrates the typical OpenVINO™ Runtime С++ API workflow:

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

2. **Include OpenVINO™ Runtime libraries** in `project/CMakeLists.txt`

   @snippet snippets/CMakeLists.txt cmake:integration_example

### Use Inference Engine API to Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the Inference Engine C++ API:   

![ie_api_use_cpp]

#### Step 1. Create Inference Engine Core 

Use the following code to create Inference Engine Core to manage available devices and read network objects:

@snippet snippets/src/main.cpp part0

#### Step 2 (Optional). Configure Input and Output of the Model

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective
    

Optionally, configure input and output of the model using the steps below:

1. Load a model to a Core object:
   - IR:
        @snippet snippets/src/main.cpp part1_1
   - ONNX:
        @snippet snippets/src/main.cpp part1_2
   - Paddle:
        @snippet snippets/src/main.cpp part1_3
   - OpenVINO Model:
        @snippet snippets/src/main.cpp part1_4_1
        @snippet snippets/src/main.cpp part1_4_2
    

2. Request input and output information using `ov::Model::inputs()`, and `ov::Model::outputs()` methods:
    @snippet snippets/src/main.cpp part2
    @snippet snippets/src/main.cpp part3
   To apply some pre-post processing please read this article about [OpenVINO™ Runtime PrePostProcessor].

@sphinxdirective
.. raw:: html

    </div>
@endsphinxdirective

#### Step 3. Compile the Model

Compile the model to the device using `ov::Core::compile_model()`:

   - IR:
        @snippet snippets/src/main.cpp part4_1
   - ONNX:
        @snippet snippets/src/main.cpp part4_2
   - Paddle:
        @snippet snippets/src/main.cpp part4_3
   - OpenVINO Model:
        @snippet snippets/src/main.cpp part4_4


It creates a compiled model from a model object. The compiled model is associated with single hardware device.
It is possible to create as many networks as needed and to use them simultaneously (up to the limitation of the hardware resources).

Third parameter is a configuration for plugin. It is map of pairs: (parameter name, parameter value). Choose device from
[Supported devices](supported_plugins/Supported_Devices.md) page for more details about supported configuration parameters.

@snippet snippets/src/main.cpp part5

#### Step 4. Create an Inference Request

Create an infer request using the following code:

@snippet snippets/src/main.cpp part6

#### Step 5. Prepare Input 

You can use one of the following options to prepare input:

* **Optimal way for a single network.** Get tensor allocated by an infer request using `ov::InferRequest::get_tensor()` and feed an image and the input data to the tensors. In this case, input data must be aligned (resized manually) with a given blob size and have a correct color format.

   @snippet snippets/src/main.cpp part7

* **Optimal way for a cascade of networks (output of one network is input for another).** Get output tensor from the first request using `ov::InferRequest::get_tensor()` and set it as input for the second request using `ov::InferRequest::set_tensor()`.

   @snippet snippets/src/main.cpp part8

* **Optimal way to handle ROI (a ROI object located inside of input of one network is input for another).** It is possible to re-use shared input by several networks. You do not need to allocate separate input blob for a network if it processes a ROI object located inside of already allocated input of a previous network. For instance, when first network detects objects on a video frame (stored as input blob) and second network accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input blob (used by first network) by second network and just crop ROI without allocation of new memory using `ov::Tensor()` with passing of `ov::Tensor` and `ov::Coordinate` as parameters.

   @snippet snippets/src/main.cpp part9

   Make sure that shared input is kept valid during execution of each network. Otherwise, ROI blob may be corrupted if the original input blob (that ROI is cropped from) has already been rewritten.

* Allocate input blobs of the appropriate types and sizes, feed an image and the input data to the blobs, and call `ov::InferRequest::set_tensor()` to set these tensors for an infer request:

   @snippet snippets/src/main.cpp part10

A blob can be filled before and after `set_tensor()`.

> **NOTE**:
>
> * The `set_tensor()` method compares precision and layout of an input blob with the ones defined in step 3 and
> throws an exception if they do not match. It also compares a size of the input blob with input
> size of the read network. But if input was configured as resizable, you can set an input blob of
> any size (for example, any ROI blob). Input resize will be invoked automatically using resize
> algorithm configured on step 3. Similarly to the resize, color format conversions allow the color
> format of an input blob to differ from the color format of the read network. Color format
> conversion will be invoked automatically using color format configured on step 3.
>
> * `get_tensor()` logic is the same for pre-processable and not pre-processable input. Even if it is
> called with input configured as resizable or as having specific color format, a blob allocated by
> an infer request is returned. Its size and color format are already consistent with the
> corresponding values of the read network. No pre-processing will happen for this blob. If you
> call `get_tensor()` after `set_tensor()`, you will get the blob you set in `set_tensor()`.

#### Step 6. Start Inference

Start inference in asynchronous or synchronous mode. Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete, the app can continue doing things on the host, while accelerator is busy.

* For synchronous inference request:
   @snippet snippets/src/main.cpp part11

* For asynchronous inference request: 
   @snippet snippets/src/main.cpp part12
  `start_async` returns immediately and starts inference without blocking main thread, `infer` blocks main thread and returns when inference is completed. Call `wait` for waiting result to become available for asynchronous request.

  There are three ways to use it:
      * `ov::InferRequest::wait_for()` - specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
      * `ov::InferRequest::wait()` - waits until inference result becomes available


Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Multiple requests for single `CompiledModel` are executed sequentially one by one in FIFO order.

While request is ongoing, all its methods except `ov::InferRequest::wait` or `ov::InferRequest::wait_for` would throw an
exception.

#### Step 7. Process the Inference Results 

Go over the output blobs and process the inference results.

@snippet snippets/src/main.cpp part13

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

# Integrate OpenVINO™ into customer application {#openvino_docs_Integrate_OV_into_customer_application}

## Integrate OpenVINO™ Runtime with Your C++ Application

@sphinxdirective
.. raw:: html

    <div id="switcher-cpp" class="switcher-anchor">C++</div>
@endsphinxdirective

The following diagram illustrates usual application development workflow:

![ie_api_flow_cpp]

Read the sections below to learn about each item.

> **NOTE**: Before start using OpenVINO™ Runtime, make sure you set all environment variables during the installation. If you did not, follow the instructions from the _Set the Environment Variables_ section in the installation guides:
> * [For Windows* 10](../install_guides/installing-openvino-windows.md)
> * [For Linux*](../install_guides/installing-openvino-linux.md)
> * [For macOS*](../install_guides/installing-openvino-macos.md)
> * To build an open source version, use the [OpenVINO™ Runtime Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

### OpenVINO™ Model Representation

Before the start, it is necessary to say several words about OpenVINO™ Model representation.
In OpenVINO™ Runtime a model is represented by the `ov::Model` class.

The `ov::Model` object stores shared pointers to `ov::op::v0::Parameter`, `ov::op::v0::Result` and `ov::op::Sink` operations that are inputs, outputs and sinks of the graph.
Sinks of the graph have no consumers and are not included in the results vector. All other operations hold each other via shared pointers: child operation holds its parent (hard link). If an operation has no consumers and it's not the `Result` or `Sink` operation
(shared pointer counter is zero), then it will be destructed and won't be accessible anymore. 

Each operation in `ov::Model` has the `std::shared_ptr<ov::Node>` type.

For details on how to build a model in OpenVINO™ Runtime, see the [Build a Model in OpenVINO™ Runtime](@ref build_model) section.

#### Shapes representation

OpenVINO™ Runtime provides two types for shape representation: 

* `ov::Shape` - Represents static (fully defined) shapes.

* `ov::PartialShape` - Represents dynamic shapes. That means that the rank or some of dimensions are dynamic (undefined). `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise the conversion raises an exception.
  `ov::PartialShape` can be converted to `ov::Shape` using the `get_shape()` method if all dimensions are static; otherwise, conversion raises an exception.

    @snippet example_ngraph_utils.cpp ov:shape

  But in most cases before getting static shape using `get_shape()` method, you need to check that shape is static.

#### Element types

`ov::element::Type` class represents 

### Link with OpenVINO™ Runtime

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

### Use OpenVINO™ Runtime API to Implement Inference Pipeline

This section provides step-by-step instructions to implement a typical inference pipeline with the OpenVINO™ Runtime C++ API:

![ie_api_use_cpp]

#### Step 1. Create OpenVINO™ Runtime Core 

Include next files to work with OpenVINO™ Runtime:

@snippet snippets/src/main.cpp include

Use the following code to create OpenVINO™ Core to manage available devices and read model objects:

@snippet snippets/src/main.cpp part0

#### Step 2 (Optional). Configure Input and Output of the Model

@sphinxdirective
.. raw:: html

    <div class="collapsible-section">
@endsphinxdirective

Optionally, OpenVINO™ Runtime allows to configure input and output of the model, please read this article about [OpenVINO™ Runtime PrePostProcessor] to understand how to do it.

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
It is possible to create as many compiled models as needed and to use them simultaneously (up to the limitation of the hardware resources).

Third parameter is a configuration for device. It is list of properties which affects device behavior.
[Supported devices](supported_plugins/Supported_Devices.md) page for more details about supported configuration parameters.

@snippet snippets/src/main.cpp part5

#### Step 4. Create an Inference Request

Create an infer request using the following code:

@snippet snippets/src/main.cpp part6

#### Step 5. Set Inputs

You can use one of the following options to prepare input:

* **Optimal way for a single model.** Get tensor allocated by an infer request using `ov::InferRequest::get_tensor()` and feed input tensor and the input data to the tensors. Input tensor's shape, element type must match specific input of the model. For cases of dynamic input shapes, read [Working with dynamic shapes].

   @snippet snippets/src/main.cpp part7

* **Optimal way for a cascade of models (output of one model is input for another).** Get output tensor from the first request using `ov::InferRequest::get_tensor()` and set it as input for the second request using `ov::InferRequest::set_tensor()`. But be careful, shared tensors across compiled models can be rewritten by the first model if the first infer request is run once again, while the second model has not started yet.

   @snippet snippets/src/main.cpp part8

* **Optimal way to handle ROI (a ROI object located inside of input of one model is input for another).** It is possible to re-use shared input by several models. You do not need to allocate separate input tensor for a model if it processes a ROI object located inside of already allocated input of a previous model. For instance, when first model detects objects on a video frame (stored as input tensor) and second model accepts detected bounding boxes (ROI inside of the frame) as input. In this case, it is allowed to re-use pre-allocated input tensor (used by first model) by second model and just crop ROI without allocation of new memory using `ov::Tensor()` with passing of `ov::Tensor` and `ov::Coordinate` as parameters.

   @snippet snippets/src/main.cpp part9

   Make sure that shared input is kept valid during execution of each model. Otherwise, ROI tensor may be corrupted if the original input tensor (that ROI is cropped from) has already been rewritten.

* Allocate input tensors of the appropriate types and sizes, feed an image and the input data to the tensors, and call `ov::InferRequest::set_tensor()` to set these tensors for an infer request:

   @snippet snippets/src/main.cpp part10

A tensor can be filled before and after `set_tensor()`.

> **NOTE**:
>
> * The `set_tensor()` method compares precision and layout of an input tensor with the ones defined in step 3 and
> throws an exception if they do not match. It also compares a size of the input tensor with input
> size of the read model. But if input was configured as resizable, you can set an input tensor of
> any size (for example, any ROI tensor). Input resize will be invoked automatically using resize
> algorithm configured on step 3. Similarly to the resize, color format conversions allow the color
> format of an input tensor to differ from the color format of the read model. Color format
> conversion will be invoked automatically using color format configured on step 3.
>
> * `get_tensor()` logic is the same for pre-processable and not pre-processable input. Even if it is
> called with input configured as resizable or as having specific color format, a tensor allocated by
> an infer request is returned. Its size and color format are already consistent with the
> corresponding values of the read model. No pre-processing will happen for this tensor. If you
> call `get_tensor()` after `set_tensor()`, you will get the tensor you set in `set_tensor()`.

#### Step 6. Start Inference

Start inference in asynchronous or synchronous mode. Async API usage can improve overall frame-rate of the application, because rather than wait for inference to complete, the app can continue doing things on the host, while accelerator is busy.

* For synchronous inference request:
   @snippet snippets/src/main.cpp part11

* For asynchronous inference request: 
   @snippet snippets/src/main.cpp part12
  `start_async` returns immediately and starts inference without blocking main thread, `infer` blocks main thread and returns when inference is completed. Call `wait` for waiting result to become available for asynchronous request.

  There are two ways to use it:
      * `ov::InferRequest::wait_for()` - specify maximum duration in milliseconds to block for. The method is blocked until the specified timeout has elapsed, or the result becomes available, whichever comes first.
      * `ov::InferRequest::wait()` - waits until inference result becomes available


Both requests are thread-safe: can be called from different threads without fearing corruption and failures.

Multiple requests for single `CompiledModel` are executed sequentially one by one in FIFO order.

While request is ongoing, all its methods except `ov::InferRequest::wait` or `ov::InferRequest::wait_for` would throw
the ov::Busy exception that request is busy with computations.

#### Step 7. Process the Inference Results 

Go over the output tensors and process the inference results.

@snippet snippets/src/main.cpp part13

### Build Your Application

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

### Run Your Application

> **NOTE**: Before running, make sure you completed **Set the Environment Variables** section in [OpenVINO Installation](../../samples/cpp/hello_nv12_input_classification/README.md) document so that the application can find the libraries.

To run compiled applications on Microsoft* Windows* OS, make sure that Microsoft* Visual C++ 2017 is installed and
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

Use the following code to create Inference Engine Core to manage available devices and read model objects: 
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

---
sidebar_position: 3
sidebar_label: "Running and Integrating Inference Pipeline"
description: "Learn how to implement a typical inference pipeline of OpenVINO™ Runtime in an application.⠀"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Running and Integrating Inference Pipeline


OpenVINO Runtime is a set of C++ libraries with C and Python bindings, providing a common
API to run inference on various devices. Each device (integrated with OpenVINO’s plugin
architecture) offers the common, as well as hardware-specific APIs for more configuration
options. Note that OpenVINO Runtime may also be integrated with other frameworks and work
as their backend, for example, using torch.compile.
The scheme below illustrates the typical workflow for deploying a trained deep learning
model in an application:

![](IMPLEMENT_PIPELINE_with_API_C.svg)

This guide will show you how to implement a typical OpenVINO™ Runtime inference pipeline
in your application. Before proceeding, check how
[model conversion](../model-preparation/convert-model-to-ir/index.md)
works in OpenVINO and how it may affect your applications’ performance. Make sure you have
installed OpenVINO Runtime and set environment variables (otherwise, the `find_package`
calls will not find OpenVINO\_DIR):

<Tabs>
  <TabItem value="lin" label="Linux">
  ```console
  <INSTALL_DIR>/setupvars.sh
  ```
  </TabItem>
  <TabItem value="win" label="Windows">
  PowerShell:

  ```console
  <INSTALL_DIR>/setupvars.sh
  ```

  Command Prompt

  ```console
  cd  <INSTALL_DIR>
  setupvars.bat
  ```
  </TabItem>
</Tabs>

## Step 1. Create OpenVINO Runtime Core

Initiate working with OpenVINO in your application by including the OpenVINO™ Runtime
components:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import openvino as ov
  ```

  ```python
  core = ov.Core()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  #include <openvino/openvino.hpp>
  ```

  ```cpp
  ov::Core core;
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  #include <openvino/c/openvino.h>
  ```

  ```cpp
  ov_core_t* core = NULL;
  ov_core_create(&core);
  ```
  </TabItem>
</Tabs>

## Step 2. Compile the Model

Compile the model with `ov::Core::compile_model()`, defining the device or mode to use
for inference. The following example uses the
[AUTO mode](inference-devices-and-modes/auto-device-selection/index.md),
which selects the device for you. To learn more about supported devices and inference modes,
see the [Inference Devices and Modes](inference-devices-and-modes/index.md)
section.

<Tabs>
  <TabItem value="py" label="Python">
  <Tabs>
    <TabItem value="ir" label="IR">
    ```python
    compiled_model = core.compile_model("model.xml", "AUTO")
    ```
    </TabItem>
    <TabItem value="onnx" label="ONNX">
    ```python
    compiled_model = core.compile_model("model.onnx", "AUTO")
    ```
    </TabItem>
    <TabItem value="paddlepaddle" label="PaddlePaddle">
    ```python
    compiled_model = core.compile_model("model.pdmodel", "AUTO")
    ```
    </TabItem>
    <TabItem value="tensorflow" label="TensorFlow">
    ```python
    compiled_model = core.compile_model("model.pb", "AUTO")
    ```
    </TabItem>
    <TabItem value="tensorflow-lite" label="TensorFlow Lite">
    ```python
    compiled_model = core.compile_model("model.tflite", "AUTO")
    ```
    </TabItem>
    <TabItem value="openvinomodel" label="ov::Model">
    ```python
    def create_model():
        # This example shows how to create ov::Function
        #
        # To construct a model, please follow
        # https://docs.openvino.ai/2026/openvino-workflow/running-inference/model-representation.html
        data = ov.opset8.parameter([3, 1, 2], ov.Type.f32)
        res = ov.opset8.result(data)
        return ov.Model([res], [data], "model")

    model = create_model()
    compiled_model = core.compile_model(model, "AUTO")
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="cpp" label="C++">
  <Tabs>
    <TabItem value="ir" label="IR">
    ```cpp
    ov::CompiledModel compiled_model = core.compile_model("model.xml", "AUTO");
    ```
    </TabItem>
    <TabItem value="onnx" label="ONNX">
    ```cpp
    ov::CompiledModel compiled_model = core.compile_model("model.onnx", "AUTO");
    ```
    </TabItem>
    <TabItem value="paddlepaddle" label="PaddlePaddle">
    ```cpp
    ov::CompiledModel compiled_model = core.compile_model("model.pdmodel", "AUTO");
    ```
    </TabItem>
    <TabItem value="tensorflow" label="TensorFlow">
    ```cpp
    ov::CompiledModel compiled_model = core.compile_model("model.pb", "AUTO");
    ```
    </TabItem>
    <TabItem value="tensorflow-lite" label="TensorFlow Lite">
    ```cpp
    ov::CompiledModel compiled_model = core.compile_model("model.tflite", "AUTO");
    ```
    </TabItem>
    <TabItem value="openvinomodel" label="ov::Model">
    ```cpp
    auto create_model = []() {
        std::shared_ptr<ov::Model> model;
        // To construct a model, please follow
        // https://docs.openvino.ai/2026/openvino-workflow/running-inference/model-representation.html
        return model;
    };
    std::shared_ptr<ov::Model> model = create_model();
    compiled_model = core.compile_model(model, "AUTO");
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="c" label="C">
  <Tabs>
    <TabItem value="ir" label="IR">
    ```cpp
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model_from_file(core, "model.xml", "AUTO", 0, &compiled_model);
    ```
    </TabItem>
    <TabItem value="onnx" label="ONNX">
    ```cpp
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model_from_file(core, "model.onnx", "AUTO", 0, &compiled_model);
    ```
    </TabItem>
    <TabItem value="paddlepaddle" label="PaddlePaddle">
    ```cpp
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model_from_file(core, "model.pdmodel", "AUTO", 0, &compiled_model);
    ```
    </TabItem>
    <TabItem value="tensorflow" label="TensorFlow">
    ```cpp
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model_from_file(core, "model.pb", "AUTO", 0, &compiled_model);
    ```
    </TabItem>
    <TabItem value="tensorflow-lite" label="TensorFlow Lite">
    ```cpp
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model_from_file(core, "model.tflite", "AUTO", 0, &compiled_model);
    ```
    </TabItem>
    <TabItem value="openvinomodel" label="ov::Model">
    ```cpp
    // Construct a model
    ov_model_t* model = NULL;
    ov_core_read_model(core, "model.xml", NULL, &model);
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model(core, model, "AUTO", 0, &compiled_model);
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

The `ov::CompiledModel` class represents a compiled model and enables you to get
information inputs or output ports by a tensor name or index. This approach is aligned with
most frameworks. The `ov::Model` object represents any models inside the OpenVINO™ Runtime.
For more details, refer to
[OpenVINO™ Model representation](model-representation.md).

## Step 3. Create an Inference Request

Use the `ov::InferRequest` class methods to create an infer request. For more details,
see the article on
[InferRequest](inference-request/index.md).

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  infer_request = compiled_model.create_infer_request()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov::InferRequest infer_request = compiled_model.create_infer_request();
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_infer_request_t* infer_request = NULL;
  ov_compiled_model_create_infer_request(compiled_model, &infer_request);
  ```
  </TabItem>
</Tabs>

## Step 4. Set Inputs

Create `ov::Tensor`, you can use external memory for that , and use the
`ov::InferRequest::set_input_tensor` method to send this tensor to the device.
For more info on textual data as input, see the
[String Tensors](model-input-output/string-tensors.md) article.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Create tensor from external memory
  input_tensor = ov.Tensor(array=memory, shared_memory=True)
  # Set input tensor for model with one input
  infer_request.set_input_tensor(input_tensor)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Get input port for model with one input
  auto input_port = compiled_model.input();
  // Create tensor from external memory
  ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), memory_ptr);
  // Set input tensor for model with one input
  infer_request.set_input_tensor(input_tensor);
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  // Get input port for model with one input
  ov_output_const_port_t* input_port = NULL;
  ov_compiled_model_input(compiled_model, &input_port);
  // Get the input shape from input port
  ov_shape_t input_shape;
  ov_const_port_get_shape(input_port, &input_shape);
  // Get the the type of input
  ov_element_type_e input_type;
  ov_port_get_element_type(input_port, &input_type);
  // Create tensor from external memory
  ov_tensor_t* tensor = NULL;
  ov_tensor_create_from_host_ptr(input_type, input_shape, memory_ptr, &tensor);
  // Set input tensor for model with one input
  ov_infer_request_set_input_tensor(infer_request, tensor);
  ```
  </TabItem>
</Tabs>

## Step 5. Start Inference

Use either `ov::InferRequest::start_async` or `ov::infer_request.infer()` to start model
inference. To learn how they work, see the
[OpenVINO Inference Request](inference-request/index.md)
article. The following example uses the asynchronous option and calls
`ov::InferRequest::wait` to wait for the inference results.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  infer_request.start_async()
  infer_request.wait()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  infer_request.start_async();
  infer_request.wait();
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_infer_request_start_async(infer_request);
  ov_infer_request_wait(infer_request);
  ```
  </TabItem>
</Tabs>

## Step 6. Process the Inference Results

Get output tensors and process the inference results.
For more info on textual data as input, see the
[String Tensors](model-input-output/string-tensors.md) article.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  # Get output tensor for model with one output
  output = infer_request.get_output_tensor()
  output_buffer = output.data
  # output_buffer[] - accessing output tensor data
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  // Get output tensor by tensor name
  auto output = infer_request.get_tensor("tensor_name");
  const float *output_buffer = output.data<const float>();
  // output_buffer[] - accessing output tensor data
  ```
  </TabItem>
  <TabItem value="c" label="C">
  ```cpp
  ov_tensor_t* output_tensor = NULL;
  // Get output tensor by tensor index
  ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor);
  ```
  </TabItem>
</Tabs>

## Step 7. [only for C] Release the allocated objects

To avoid memory leak, applications developed with the C API need to release the allocated
objects in the following order.

<Tabs>
  <TabItem value="c" label="C">
  ```cpp
  ov_shape_free(&input_shape);
  ov_tensor_free(output_tensor);
  ov_output_const_port_free(input_port);
  ov_tensor_free(tensor);
  ov_infer_request_free(infer_request);
  ov_compiled_model_free(compiled_model);
  ov_model_free(model);
  ov_core_free(core);
  ```
  </TabItem>
</Tabs>

## Build Your Application

If you have integrated OpenVINO with your application, you will need to adjust your
application build process as well. Of course, there are multiple ways this stage may be
done, so you will need to choose the one best for your project. To learn about the basics of
OpenVINO build process, refer to the
[documentation on GitHub](https://github.com/openvinotoolkit/openvino/blob/master/docs/dev/build.md).

The following example uses a C++ & C application together with CMake,
for project configuration.

1. Create Structure for project:
   
   <Tabs>
     <TabItem value="cpp" label="C++">
     ```cpp
     project/
        ├── CMakeLists.txt  - CMake file to build
        ├── ...             - Additional folders like includes/
        └── src/            - source folder
            └── main.cpp
     build/                  - build directory
        ...
     ```
     </TabItem>
     <TabItem value="c" label="C">
     ```cpp
     project/
        ├── CMakeLists.txt  - CMake file to build
        ├── ...             - Additional folders like includes/
        └── src/            - source folder
            └── main.c
     build/                  - build directory
        ...
     ```
     </TabItem>
   </Tabs>
2. Configure the CMake build
   
   For details on additional CMake build options, refer to the
   [CMake page](https://cmake.org/cmake/help/latest/manual/cmake.1.html#manual:cmake(1)).
   
   <Tabs>
     <TabItem value="cpp" label="C++">
     ```cpp
     cmake_minimum_required(VERSION 3.10)
     set(CMAKE_CXX_STANDARD 17)

     find_package(OpenVINO REQUIRED)

     add_executable(${TARGET_NAME} src/main.cpp)

     target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)
     ```
     </TabItem>
     <TabItem value="c" label="C">
     ```cpp
     cmake_minimum_required(VERSION 3.10)
     set(CMAKE_CXX_STANDARD 17)

     find_package(OpenVINO REQUIRED)

     add_executable(${TARGET_NAME_C} src/main.c)

     target_link_libraries(${TARGET_NAME_C} PRIVATE openvino::runtime::c)
     ```
     </TabItem>
     <TabItem value="cpp-2" label="C++ (PyPI)">
     ```cpp
     cmake_minimum_required(VERSION 3.10)
     set(CMAKE_CXX_STANDARD 17)

     if(NOT CMAKE_CROSSCOMPILING)
         find_package(Python3 QUIET COMPONENTS Interpreter)
         if(Python3_Interpreter_FOUND)
             execute_process(
                 COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
                 OUTPUT_VARIABLE OpenVINO_DIR_PY
                 ERROR_QUIET)
         endif()
     endif()

     find_package(OpenVINO REQUIRED PATHS "${OpenVINO_DIR_PY}")

     add_executable(${TARGET_NAME_PY} src/main.cpp)

     target_link_libraries(${TARGET_NAME_PY} PRIVATE openvino::runtime)
     ```
     </TabItem>
   </Tabs>
3. Build Project
   
   Use CMake to build the project on your system:
   
   ```bash
   cd build/
   cmake ../project
   cmake --build .
   ```

## Additional Resources

- To see working implementation of the steps, check out the
  [Learn OpenVINO](../../get-started/learn-openvino/index.md) section, including
  <a href="./../../notebooks/openvino-api-with-output.html">OpenVINO™ Runtime API Tutorial</a>.
- Models in the OpenVINO IR format on [Hugging Face](https://huggingface.co/models).
- [Using Encrypted Models with OpenVINO](../../documentation/openvino-security.md#encrypted-models)

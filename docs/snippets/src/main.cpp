// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [include]
#include <openvino/openvino.hpp>
//! [include]

int main() {
//! [part1]
ov::Core core;
//! [part1]

ov::CompiledModel compiled_model;
{
//! [part2_1]
ov::CompiledModel compiled_model = core.compile_model("model.xml", "AUTO");
//! [part2_1]
}
{
//! [part2_2]
ov::CompiledModel compiled_model = core.compile_model("model.onnx", "AUTO");
//! [part2_2]
}
{
//! [part2_3]
ov::CompiledModel compiled_model = core.compile_model("model.pdmodel", "AUTO");
//! [part2_3]
}
{
//! [part2_4]
ov::CompiledModel compiled_model = core.compile_model("model.pb", "AUTO");
//! [part2_4]
}
{
//! [part2_5]
ov::CompiledModel compiled_model = core.compile_model("model.tflite", "AUTO");
//! [part2_5]
}
{
//! [part2_6]
auto create_model = []() {
    std::shared_ptr<ov::Model> model;
    // To construct a model, please follow
    // https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html
    return model;
};
std::shared_ptr<ov::Model> model = create_model();
compiled_model = core.compile_model(model, "AUTO");
//! [part2_6]
}

//! [part3]
ov::InferRequest infer_request = compiled_model.create_infer_request();
//! [part3]

void * memory_ptr = nullptr;
//! [part4]
// Get input port for model with one input
auto input_port = compiled_model.input();
// Create tensor from external memory
ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), memory_ptr);
// Set input tensor for model with one input
infer_request.set_input_tensor(input_tensor);
//! [part4]

//! [part5]
infer_request.start_async();
infer_request.wait();
//! [part5]

//! [part6]
// Get output tensor by tensor name
auto output = infer_request.get_tensor("tensor_name");
const float *output_buffer = output.data<const float>();
// output_buffer[] - accessing output tensor data
//! [part6]
return 0;
}
/*
//! [part7]
project/
   ├── CMakeLists.txt  - CMake file to build
   ├── ...             - Additional folders like includes/
   └── src/            - source folder
       └── main.cpp
build/                  - build directory
   ...

//! [part7]
*/

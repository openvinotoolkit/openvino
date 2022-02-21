// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [include]
#include <openvino/openvino.hpp>
//! [include]

//! [part1_4_1]
std::shared_ptr<ov::Model> create_model() {
    std::shared_ptr<ov::Model> model;
    // To construct a network, please follow 
    // https://docs.openvino.ai/latest/openvino_docs_OV_Runtime_UG_Model_Representation.html
    return model;
}
//! [part1_4_1]

int main() {
//! [part1]
ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiled_model;
//! [part1]

//! [part2_1]
compiled_model = core.compile_model("model.xml", "AUTO");
//! [part2_1]
//! [part2_2]
compiled_model = core.compile_model("model.onnx", "AUTO");
//! [part2_2]
//! [part2_3]
compiled_model = core.compile_model("model.pdmodel", "AUTO");
//! [part2_3]
//! [part2_4]
compiled_model = core.compile_model(model, "AUTO");
//! [part2_4]

//! [part3]
auto infer_request = compiled_model.create_infer_request();
//! [part3]

void * memory_ptr = nullptr;
//! [part4]
// Get input port for model with one input
auto input_port = model->input();
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
// Get output tensor for model with one output
auto output = infer_request.get_output_tensor();
const float *output_buffer = output.data<const float>();
/* output_buffer[] - accessing output tensor data */
//! [part6]
return 0;
}

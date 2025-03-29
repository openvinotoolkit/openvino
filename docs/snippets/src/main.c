// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [include]
#include <openvino/c/openvino.h>
//! [include]

int main() {
//! [part1]
ov_core_t* core = NULL;
ov_core_create(&core);
//! [part1]

{
//! [part2_1]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model_from_file(core, "model.xml", "AUTO", 0, &compiled_model);
//! [part2_1]
}
{
//! [part2_2]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model_from_file(core, "model.onnx", "AUTO", 0, &compiled_model);
//! [part2_2]
}
{
//! [part2_3]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model_from_file(core, "model.pdmodel", "AUTO", 0, &compiled_model);
//! [part2_3]
}
{
//! [part2_4]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model_from_file(core, "model.pb", "AUTO", 0, &compiled_model);
//! [part2_4]
}
{
//! [part2_5]
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model_from_file(core, "model.tflite", "AUTO", 0, &compiled_model);
//! [part2_5]
}

//! [part2_6]
// Construct a model
ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model(core, model, "AUTO", 0, &compiled_model);
//! [part2_6]


//! [part3]
ov_infer_request_t* infer_request = NULL;
ov_compiled_model_create_infer_request(compiled_model, &infer_request);
//! [part3]

void * memory_ptr = NULL;
//! [part4]
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
//! [part4]

//! [part5]
ov_infer_request_start_async(infer_request);
ov_infer_request_wait(infer_request);
//! [part5]

//! [part6]
ov_tensor_t* output_tensor = NULL;
// Get output tensor by tensor index
ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor);
//! [part6]

//! [part8]
ov_shape_free(&input_shape);
ov_tensor_free(output_tensor);
ov_output_const_port_free(input_port);
ov_tensor_free(tensor);
ov_infer_request_free(infer_request);
ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_core_free(core);
//! [part8]
return 0;
}
/*
//! [part7]
project/
   ├── CMakeLists.txt  - CMake file to build
   ├── ...             - Additional folders like includes/
   └── src/            - source folder
       └── main.c
build/                  - build directory
   ...

//! [part7]
*/

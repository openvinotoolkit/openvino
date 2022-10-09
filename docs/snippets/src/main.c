// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//! [include]
#include <openvino/c/openvino.h>
//! [include]

int main() {

//! [part1]
ov_core_t* core = NULL; // need to free by ov_core_free(core)
ov_core_create(&core);
//! [part1]

const char* device_name = "CPU";
{
//! [part2_1]
ov_compiled_model_t* compiled_model = NULL; // need to free by ov_compiled_model_free(compiled_model)
ov_core_compile_model_from_file(core, "model.xml", device_name, 0, &compiled_model);
//! [part2_1]
}
{
//! [part2_2]
ov_compiled_model_t* compiled_model = NULL; // need to free by ov_compiled_model_free(compiled_model)
ov_core_compile_model_from_file(core, "model.onnx", device_name, 0, &compiled_model);
//! [part2_2]
}
{
//! [part2_3]
ov_compiled_model_t* compiled_model = NULL; // need to free by ov_compiled_model_free(compiled_model)
ov_core_compile_model_from_file(core, "model.pdmodel", device_name, 0, &compiled_model);
//! [part2_3]
}

//! [part2_4]
// Construct a model
ov_model_t* model = NULL; // need to free by ov_model_free(model)
ov_core_read_model(core, "model.xml", NULL, &model);
ov_compiled_model_t* compiled_model = NULL; // need to free by ov_compiled_model_free(compiled_model)
ov_core_compile_model(core, model, device_name, 0, &compiled_model);
//! [part2_4]


//! [part3]
ov_infer_request_t* infer_request = NULL; // need to free by ov_infer_request_free(infer_request)
ov_compiled_model_create_infer_request(compiled_model, &infer_request);
//! [part3]

ov_shape_t input_shape;
void* img_data = NULL;
ov_element_type_e input_type = U8;
//! [part4]
ov_tensor_t* tensor = NULL; // need to free by ov_tensor_free(tensor)
// Create tensor from external memory
ov_tensor_create_from_host_ptr(input_type, input_shape, img_data, &tensor);
// Set input tensor for model with one input
ov_infer_request_set_input_tensor_by_index(infer_request, 0, tensor);
//! [part4]

//! [part5]
ov_infer_request_infer(infer_request);
//! [part5]

//! [part6]
ov_tensor_t* output_tensor = NULL; // need to free by ov_tensor_free(output_tensor)
// Get output tensor by tensor index
ov_infer_request_get_output_tensor_by_index(infer_request, 0, &output_tensor);
//! [part6]

//! [part8]
ov_tensor_free(output_tensor);
ov_tensor_free(tensor);
ov_infer_request_free(infer_request);
ov_compiled_model_free(compiled_model);
ov_core_free(core);
//! [part8]


return 0;
}

//! [part7]
// Create a structure for the project:
project/
   ├── CMakeLists.txt  - CMake file to build
   ├── ...             - Additional folders like includes/
   └── src/            - source folder
       └── main.c
build/                  - build directory
   ...      

//! [part7]
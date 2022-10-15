// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/c/openvino.h>

void reshape_with_dynamics() {
{
//! [ov_dynamic_shapes:reshape_undefined]
ov_core_t* core = NULL;
ov_core_create(&core);

ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);

// Set one static dimension (= 1) and another dynamic dimension (= Dimension())
{
ov_partial_shape_t partial_shape;
int64_t rank = 2;
ov_dimension_t dims[2] = {{1, 1}, {-1, -1}};
ov_partial_shape_create(rank, dims, &partial_shape);
ov_model_reshape(model, "tensor_name", &partial_shape, 1); // {1,?}
}

// Or set both dimensions as dynamic if both are going to be changed dynamically
{
ov_partial_shape_t partial_shape;
int64_t rank = 2;
ov_dimension_t dims[2] = {{-1, -1}, {-1, -1}};
ov_partial_shape_create(rank, dims, &partial_shape);
ov_model_reshape(model, "tensor_name", &partial_shape, 1); // {?,?}
}
//! [ov_dynamic_shapes:reshape_undefined]

//! [ov_dynamic_shapes:reshape_bounds]
// Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
{
ov_partial_shape_t partial_shape;
int64_t rank = 2;
ov_dimension_t dims[2] = {{1, 10}, {8, 512}};
ov_partial_shape_create(rank, dims, &partial_shape);
ov_model_reshape(model, "tensor_name", &partial_shape, 1); // {1..10,8..512}
}

// Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
{
ov_partial_shape_t partial_shape;
int64_t rank = 2;
ov_dimension_t dims[2] = {{-1, -1}, {8, 512}};
ov_partial_shape_create(rank, dims, &partial_shape);
ov_model_reshape(model, "tensor_name", &partial_shape, 1); // {?,8..512}
}
//! [ov_dynamic_shapes:reshape_bounds]
}

{
ov_core_t* core = NULL;
ov_core_create(&core);
ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);
//! [ov_dynamic_shapes:print_dynamic]
// Print output partial shape
ov_output_port_t* output_port = NULL;
ov_model_output(model, &output_port);
ov_partial_shape_t partial_shape;
ov_port_get_partial_shape(output_port, &partial_shape);
char * str_partial_shape = ov_partial_shape_to_string(partial_shape);
printf("The output partial shape: %s", str_partial_shape);

// Print input partial shape
ov_output_port_t* input_port = NULL;
ov_model_input(model, &input_port);
ov_partial_shape_t partial_shape;
ov_port_get_partial_shape(input_port, &partial_shape);
char * str_partial_shape = ov_partial_shape_to_string(partial_shape);
printf("The input partial shape: %s", str_partial_shape);
//! [ov_dynamic_shapes:print_dynamic]
}

{
ov_core_t* core = NULL;
ov_core_create(&core);

//! [ov_dynamic_shapes:detect_dynamic]
ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);

ov_output_port_t* input_port = NULL;
ov_model_input_by_index(model, 0, &input_port);
ov_partial_shape_t partial_shape;
ov_port_get_partial_shape(input_port, &partial_shape);
if (ov_partial_shape_is_dynamic(partial_shape)) {
    // input is dynamic
}

ov_output_port_t* output_port = NULL;
ov_model_output_by_index(model, 0, &output_port);
ov_partial_shape_t partial_shape;
ov_port_get_partial_shape(output_port, &partial_shape);
if (ov_partial_shape_is_dynamic(partial_shape)) {
    // output is dynamic
}
//! [ov_dynamic_shapes:detect_dynamic]
}

}

void set_tensor() {
ov_core_t* core = NULL;
ov_core_create(&core);

ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);
const char* device_name = "CPU";
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model(core, model, device_name, 0, &compiled_model);

ov_infer_request_t* infer_request = NULL;
ov_compiled_model_create_infer_request(compiled_model, &infer_request);

//! [ov_dynamic_shapes:set_input_tensor]
// The first inference call

// Create tensor compatible with the model input
// Shape {1, 128} is compatible with any reshape statements made in previous examples
ov_output_port_t* port = NULL;
ov_model_input(model, &port);
ov_element_type_e* type = NULL;
ov_port_get_element_type(port, type);

ov_shape_t input_shape;
int64_t rank = 2;
int64_t dims[2] = {1, 128};
ov_shape_create(rank, dims, &input_shape);

ov_tensor_t* input_tensor_1 = NULL;
ov_tensor_create(type, input_shape, &input_tensor_1);
// ... write values to input_tensor_1

// Set the tensor as an input for the infer request
ov_infer_request_set_input_tensor(infer_request, input_tensor_1);

// Do the inference
ov_infer_request_infer(infer_request);

// Retrieve a tensor representing the output data
ov_tensor_t* output_tensor = NULL;
ov_infer_request_get_output_tensor(infer_request, &output_tensor);

// For dynamic models output shape usually depends on input shape,
// that means shape of output tensor is initialized after the first inference only
// and has to be queried after every infer request
ov_shape_t output_shape_1;
ov_tensor_get_shape(output_tensor, &output_shape_1);

// Take a pointer of an appropriate type to tensor data and read elements according to the shape
// Assuming model output is f32 data type
void* data_1 = NULL;
ov_tensor_data(output_tensor, &data_1);
// ... read values

// The second inference call, repeat steps:

// Create another tensor (if the previous one cannot be utilized)
// Notice, the shape is different from input_tensor_1
ov_shape_t input_shape;
int64_t rank = 2;
int64_t dims[2] = {1, 200};
ov_shape_create(rank, dims, &input_shape);

ov_tensor_t* input_tensor_2 = NULL;
ov_tensor_create(type, input_shape, &input_tensor_2);
// ... write values to input_tensor_2

ov_infer_request_set_input_tensor(infer_request, input_tensor_2);

ov_infer_request_infer(infer_request);

// No need to call infer_request.get_output_tensor() again
// output_tensor queried after the first inference call above is valid here.
// But it may not be true for the memory underneath as shape changed, so re-take a pointer:
ov_infer_request_get_output_tensor(infer_request, &output_tensor);
void* data_2 = NULL;
ov_tensor_data(output_tensor, &data_2);

// and new shape as well
ov_shape_t output_shape_2;
ov_tensor_get_shape(output_tensor, &output_shape_2);

// ... read values in data_2 according to the shape output_shape_2
//! [ov_dynamic_shapes:set_input_tensor]
}

void get_tensor() {
ov_core_t* core = NULL;
ov_core_create(&core);

ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);
const char* device_name = "CPU";
ov_compiled_model_t* compiled_model = NULL;
ov_core_compile_model(core, model, device_name, 0, &compiled_model);

ov_infer_request_t* infer_request = NULL;
ov_compiled_model_create_infer_request(compiled_model, &infer_request);

//! [ov_dynamic_shapes:get_input_tensor]
// The first inference call

// Get the tensor; shape is not initialized
ov_tensor_t* input_tensor = NULL;
ov_infer_request_get_input_tensor(infer_request, &input_tensor);

// Set shape is required
ov_shape_t input_shape;
int64_t rank = 2;
int64_t dims[2] = {1, 128};
ov_shape_create(rank, dims, &input_shape);
ov_tensor_set_shape(input_tensor, input_shape);
// ... write values to input_tensor

ov_infer_request_infer(infer_request);
ov_tensor_t* output_tensor = NULL;
ov_infer_request_get_output_tensor(infer_request, &output_tensor);
ov_shape_t output_shape_1;
ov_tensor_get_shape(output_tensor, &output_shape_1);
void* data_1 = NULL;
ov_tensor_data(output_tensor, &data_1);
// ... read values

// The second inference call, repeat steps:

// Set a new shape, may reallocate tensor memory
int64_t rank = 2;
int64_t dims[2] = {1, 200};
ov_shape_create(rank, dims, &input_shape);
ov_tensor_set_shape(input_tensor, input_shape);
// ... write values to input_tensor memory

ov_infer_request_infer(infer_request);
ov_infer_request_get_output_tensor(infer_request, &output_tensor);
ov_shape_t output_shape_2;
ov_tensor_get_shape(output_tensor, &output_shape_2);
void* data_2 = NULL;
ov_tensor_data(output_tensor, &data_2);
// ... read values in data_2 according to the shape output_shape_2
//! [ov_dynamic_shapes:get_input_tensor]
}

int main() {
reshape_with_dynamics();
get_tensor();
set_tensor();
return 0;
}

// Copyright (C) 2018-2025 Intel Corporation
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

// Set first dimension as dynamic ({-1, -1}) and remaining dimensions as static
{
ov_partial_shape_t partial_shape;
ov_dimension_t dims[4] = {{-1, -1}, {3, 3}, {224, 224}, {224, 224}};
ov_partial_shape_create(4, dims, &partial_shape);
ov_model_reshape_single_input(model, partial_shape); // {?,3,224,224}
ov_partial_shape_free(&partial_shape);
}

// Or, set third and fourth dimensions as dynamic
{
ov_partial_shape_t partial_shape;
ov_dimension_t dims[4] = {{1, 1}, {3, 3}, {-1, -1}, {-1, -1}};
ov_partial_shape_create(4, dims, &partial_shape);
ov_model_reshape_single_input(model, partial_shape); // {1,3,?,?}
ov_partial_shape_free(&partial_shape);
}
//! [ov_dynamic_shapes:reshape_undefined]

//! [ov_dynamic_shapes:reshape_bounds]
// Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
{
ov_partial_shape_t partial_shape;
ov_dimension_t dims[2] = {{1, 10}, {8, 512}};
ov_partial_shape_create(2, dims, &partial_shape);
ov_model_reshape_single_input(model, partial_shape); // {1..10,8..512}
ov_partial_shape_free(&partial_shape);
}

// Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
{
ov_partial_shape_t partial_shape;
ov_dimension_t dims[2] = {{-1, -1}, {8, 512}};
ov_partial_shape_create(2, dims, &partial_shape);
ov_model_reshape_single_input(model, partial_shape); // {?,8..512}
ov_partial_shape_free(&partial_shape);
}
//! [ov_dynamic_shapes:reshape_bounds]
ov_model_free(model);
ov_core_free(core);
}

{
ov_core_t* core = NULL;
ov_core_create(&core);
ov_model_t* model = NULL;
ov_core_read_model(core, "model.xml", NULL, &model);

//! [ov_dynamic_shapes:print_dynamic]
ov_output_const_port_t* output_port = NULL;
ov_output_const_port_t* input_port = NULL;
ov_partial_shape_t partial_shape;
const char * str_partial_shape = NULL;

// Print output partial shape
{
ov_model_const_output(model, &output_port);
ov_port_get_partial_shape(output_port, &partial_shape);
str_partial_shape = ov_partial_shape_to_string(partial_shape);
printf("The output partial shape: %s", str_partial_shape);
}

// Print input partial shape
{
ov_model_const_input(model, &input_port);
ov_port_get_partial_shape(input_port, &partial_shape);
str_partial_shape = ov_partial_shape_to_string(partial_shape);
printf("The input partial shape: %s", str_partial_shape);
}

// free allocated resource
ov_free(str_partial_shape);
ov_partial_shape_free(&partial_shape);
ov_output_const_port_free(output_port);
ov_output_const_port_free(input_port);
//! [ov_dynamic_shapes:print_dynamic]
ov_model_free(model);
ov_core_free(core);
}

{
ov_core_t* core = NULL;
ov_core_create(&core);

//! [ov_dynamic_shapes:detect_dynamic]
ov_model_t* model = NULL;
ov_output_const_port_t* input_port = NULL;
ov_output_const_port_t* output_port = NULL;
ov_partial_shape_t partial_shape;

ov_core_read_model(core, "model.xml", NULL, &model);

// for input
{
ov_model_const_input_by_index(model, 0, &input_port);
ov_port_get_partial_shape(input_port, &partial_shape);
if (ov_partial_shape_is_dynamic(partial_shape)) {
    // input is dynamic
}
}

// for output
{
ov_model_const_output_by_index(model, 0, &output_port);
ov_port_get_partial_shape(output_port, &partial_shape);
if (ov_partial_shape_is_dynamic(partial_shape)) {
    // output is dynamic
}
}

// free allocated resource
ov_partial_shape_free(&partial_shape);
ov_output_const_port_free(input_port);
ov_output_const_port_free(output_port);
//! [ov_dynamic_shapes:detect_dynamic]
ov_model_free(model);
ov_core_free(core);
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
ov_output_const_port_t* input_port = NULL;
ov_element_type_e type = DYNAMIC;
ov_shape_t input_shape_1;
ov_tensor_t* input_tensor_1 = NULL;
ov_tensor_t* output_tensor = NULL;
ov_shape_t output_shape_1;
void* data_1 = NULL;
ov_shape_t input_shape_2;
ov_tensor_t* input_tensor_2 = NULL;
ov_shape_t output_shape_2;
void* data_2 = NULL;
// The first inference call

// Create tensor compatible with the model input
// Shape {1, 128} is compatible with any reshape statements made in previous examples
{
ov_model_const_input(model, &input_port);
ov_port_get_element_type(input_port, &type);
int64_t dims[2] = {1, 128};
ov_shape_create(2, dims, &input_shape_1);
ov_tensor_create(type, input_shape_1, &input_tensor_1);
// ... write values to input_tensor
}

// Set the tensor as an input for the infer request
ov_infer_request_set_input_tensor(infer_request, input_tensor_1);

// Do the inference
ov_infer_request_infer(infer_request);

// Retrieve a tensor representing the output data
ov_infer_request_get_output_tensor(infer_request, &output_tensor);

// For dynamic models output shape usually depends on input shape,
// that means shape of output tensor is initialized after the first inference only
// and has to be queried after every infer request
ov_tensor_get_shape(output_tensor, &output_shape_1);

// Take a pointer of an appropriate type to tensor data and read elements according to the shape
// Assuming model output is f32 data type
ov_tensor_data(output_tensor, &data_1);
// ... read values

// The second inference call, repeat steps:

// Create another tensor (if the previous one cannot be utilized)
// Notice, the shape is different from input_tensor_1
{
int64_t dims[2] = {1, 200};
ov_shape_create(2, dims, &input_shape_2);
ov_tensor_create(type, input_shape_2, &input_tensor_2);
// ... write values to input_tensor_2
}

ov_infer_request_set_input_tensor(infer_request, input_tensor_2);
ov_infer_request_infer(infer_request);

// No need to call infer_request.get_output_tensor() again
// output_tensor queried after the first inference call above is valid here.
// But it may not be true for the memory underneath as shape changed, so re-take a pointer:
ov_tensor_data(output_tensor, &data_2);

// and new shape as well
ov_tensor_get_shape(output_tensor, &output_shape_2);
// ... read values in data_2 according to the shape output_shape_2

// free resource
ov_output_const_port_free(input_port);
ov_shape_free(&input_shape_1);
ov_tensor_free(input_tensor_1);
ov_shape_free(&output_shape_1);
ov_shape_free(&input_shape_2);
ov_tensor_free(input_tensor_2);
ov_shape_free(&output_shape_2);
ov_tensor_free(output_tensor);

//! [ov_dynamic_shapes:set_input_tensor]
ov_infer_request_free(infer_request);
ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_core_free(core);
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
ov_tensor_t* input_tensor = NULL;
ov_shape_t input_shape_1;
ov_tensor_t* output_tensor = NULL;
void* data_1 = NULL;
ov_shape_t output_shape_1;
ov_shape_t input_shape_2;
ov_shape_t output_shape_2;
void* data_2 = NULL;
// The first inference call
// Get the tensor; shape is not initialized
ov_infer_request_get_input_tensor(infer_request, &input_tensor);

// Set shape is required
{
int64_t dims[2] = {1, 128};
ov_shape_create(2, dims, &input_shape_1);
ov_tensor_set_shape(input_tensor, input_shape_1);
// ... write values to input_tensor
}
// do inference
ov_infer_request_infer(infer_request);
// get output tensor data & shape
{
ov_infer_request_get_output_tensor(infer_request, &output_tensor);
ov_tensor_get_shape(output_tensor, &output_shape_1);
ov_tensor_data(output_tensor, &data_1);
// ... read values
}

// The second inference call, repeat steps:
// Set a new shape, may reallocate tensor memory
{
int64_t dims[2] = {1, 200};
ov_shape_create(2, dims, &input_shape_2);
ov_tensor_set_shape(input_tensor, input_shape_2);
// ... write values to input_tensor memory
}
// do inference
ov_infer_request_infer(infer_request);
// get output tensor data & shape
{
ov_tensor_get_shape(output_tensor, &output_shape_2);
ov_tensor_data(output_tensor, &data_2);
// ... read values in data_2 according to the shape output_shape_2
}

ov_shape_free(&input_shape_1);
ov_shape_free(&output_shape_1);
ov_shape_free(&input_shape_2);
ov_shape_free(&output_shape_2);
ov_tensor_free(output_tensor);
//! [ov_dynamic_shapes:get_input_tensor]
ov_tensor_free(input_tensor);
ov_infer_request_free(infer_request);
ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_core_free(core);
}

int main() {
reshape_with_dynamics();
get_tensor();
set_tensor();
return 0;
}

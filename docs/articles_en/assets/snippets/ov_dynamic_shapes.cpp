// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/core/core.hpp>
#include <openvino/runtime/runtime.hpp>

void reshape_with_dynamics() {
{
//! [ov_dynamic_shapes:reshape_undefined]
ov::Core core;
auto model = core.read_model("model.xml");

// Set first dimension as dynamic (ov::Dimension()) and remaining dimensions as static
model->reshape({{ov::Dimension(), 3, 224, 224}});  // {?,3,224,224}

// Or, set third and fourth dimensions as dynamic
model->reshape({{1, 3, ov::Dimension(), ov::Dimension()}});  // {1,3,?,?}
//! [ov_dynamic_shapes:reshape_undefined]
//! [ov_dynamic_shapes:reshape_bounds]
// Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
model->reshape({{ov::Dimension(1, 10), ov::Dimension(8, 512)}});  // {1..10,8..512}

// Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
model->reshape({{-1, ov::Dimension(8, 512)}});   // {?,8..512}
//! [ov_dynamic_shapes:reshape_bounds]
}
{
ov::Core core;
auto model = core.read_model("model.xml");
//! [ov_dynamic_shapes:print_dynamic]
// Print output partial shape
std::cout << model->output().get_partial_shape() << "\n";

// Print input partial shape
std::cout << model->input().get_partial_shape() << "\n";
//! [ov_dynamic_shapes:print_dynamic]
}
{
ov::Core core;
//! [ov_dynamic_shapes:detect_dynamic]
auto model = core.read_model("model.xml");

if (model->input(0).get_partial_shape().is_dynamic()) {
    // input is dynamic
}

if (model->output(0).get_partial_shape().is_dynamic()) {
    // output is dynamic
}

if (model->output(0).get_partial_shape()[1].is_dynamic()) {
    // 1-st dimension of output is dynamic
}
//! [ov_dynamic_shapes:detect_dynamic]
}
{
//! [ov_dynamic_shapes:check_inputs]
ov::Core core;
auto model = core.read_model("model.xml");

// Print info of first input layer
std::cout << model->input(0).get_partial_shape() << "\n";

// Print info of second input layer
std::cout << model->input(1).get_partial_shape() << "\n";

//etc
//! [ov_dynamic_shapes:check_inputs]
}
{
ov::Core core;
auto model = core.read_model("model.xml");
//! [ov_dynamic_shapes:reshape_multiple_inputs]
// Assign dynamic shapes to second dimension in every input layer
std::map<ov::Output<ov::Node>, ov::PartialShape> port_to_shape;
for (const ov::Output<ov::Node>& input : model->inputs()) {
    ov::PartialShape shape = input.get_partial_shape();
    shape[1] = -1;
    port_to_shape[input] = shape;
}
model->reshape(port_to_shape);
//! [ov_dynamic_shapes:reshape_multiple_inputs]
}
}

void set_tensor() {
ov::Core core;
auto model = core.read_model("model.xml");
auto executable = core.compile_model(model);
auto infer_request = executable.create_infer_request();
//! [ov_dynamic_shapes:set_input_tensor]
// The first inference call

// Create tensor compatible with the model input
// Shape {1, 128} is compatible with any reshape statements made in previous examples
auto input_tensor_1 = ov::Tensor(model->input().get_element_type(), {1, 128});
// ... write values to input_tensor_1

// Set the tensor as an input for the infer request
infer_request.set_input_tensor(input_tensor_1);

// Do the inference
infer_request.infer();

// Retrieve a tensor representing the output data
ov::Tensor output_tensor = infer_request.get_output_tensor();

// For dynamic models output shape usually depends on input shape,
// that means shape of output tensor is initialized after the first inference only
// and has to be queried after every infer request
auto output_shape_1 = output_tensor.get_shape();

// Take a pointer of an appropriate type to tensor data and read elements according to the shape
// Assuming model output is f32 data type
auto data_1 = output_tensor.data<float>();
// ... read values

// The second inference call, repeat steps:

// Create another tensor (if the previous one cannot be utilized)
// Notice, the shape is different from input_tensor_1
auto input_tensor_2 = ov::Tensor(model->input().get_element_type(), {1, 200});
// ... write values to input_tensor_2

infer_request.set_input_tensor(input_tensor_2);

infer_request.infer();

// No need to call infer_request.get_output_tensor() again
// output_tensor queried after the first inference call above is valid here.
// But it may not be true for the memory underneath as shape changed, so re-take a pointer:
auto data_2 = output_tensor.data<float>();

// and new shape as well
auto output_shape_2 = output_tensor.get_shape();

// ... read values in data_2 according to the shape output_shape_2
//! [ov_dynamic_shapes:set_input_tensor]
}

void get_tensor() {
ov::Core core;
auto model = core.read_model("model.xml");
auto executable = core.compile_model(model);
auto infer_request = executable.create_infer_request();
//! [ov_dynamic_shapes:get_input_tensor]
// The first inference call

// Get the tensor; shape is not initialized
auto input_tensor = infer_request.get_input_tensor();

// Set shape is required
input_tensor.set_shape({1, 128});
// ... write values to input_tensor

infer_request.infer();
ov::Tensor output_tensor = infer_request.get_output_tensor();
auto output_shape_1 = output_tensor.get_shape();
auto data_1 = output_tensor.data<float>();
// ... read values

// The second inference call, repeat steps:

// Set a new shape, may reallocate tensor memory
input_tensor.set_shape({1, 200});
// ... write values to input_tensor memory

infer_request.infer();
auto data_2 = output_tensor.data<float>();
auto output_shape_2 = output_tensor.get_shape();
// ... read values in data_2 according to the shape output_shape_2
//! [ov_dynamic_shapes:get_input_tensor]
}

int main() {
reshape_with_dynamics();
get_tensor();
set_tensor();
return 0;
}

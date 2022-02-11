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
const std::string output_name = "output_name";
const std::string input_name = "input_name";
//! [part0]
ov::Core core;
std::shared_ptr<ov::Model> model;
ov::CompiledModel compiled_model;
//! [part0]

//! [part1_1]
model = core.read_model("model.xml");
//! [part1_1]

//! [part1_2]
model = core.read_model("model.onnx");
//! [part1_2]

//! [part1_3]
model = core.read_model("model.pdmodel");
//! [part1_3]

//! [part1_4_2]
model = create_model();
//! [part1_4_2]

//! [part2]
/** Take information about all topology inputs **/
auto inputs = model->inputs();
/** Take information about all topology outputs **/
auto outputs = model->outputs();
//! [part2]

//! [part3]
/** Iterate over all input info**/
for (auto &item : inputs) {
    // ...
}
/** Iterate over all output info**/
for (auto &item : outputs) {
    // ...
}
//! [part3]

//! [part4_1]
compiled_model = core.compile_model("model.xml", "CPU");
//! [part4_1]
//! [part4_2]
compiled_model = core.compile_model("model.onnx", "CPU");
//! [part4_2]
//! [part4_3]
compiled_model = core.compile_model("model.pdmodel", "CPU");
//! [part4_3]
//! [part4_4]
compiled_model = core.compile_model(model, "CPU");
//! [part4_4]

//! [part5]
/** Optional config. E.g. this enables profiling of performance counters. **/
ov::AnyMap config = {ov::enable_profiling(true)};
compiled_model = core.compile_model(model, "CPU", config);
//! [part5]

//! [part6]
auto infer_request = compiled_model.create_infer_request();
//! [part6]

auto infer_request1 = compiled_model.create_infer_request();
auto infer_request2 = compiled_model.create_infer_request();

//! [part7]
/** Iterate over all input tensors **/
for (auto & item : inputs) {
    /** Get input tensor **/
    auto input = infer_request.get_tensor(item.get_any_name());
    /** Fill input tensor with planes. First b channel, then g and r channels **/
//     ...
}
//! [part7]

//! [part8]
auto output = infer_request1.get_tensor(output_name);
infer_request2.set_tensor(input_name, output);
//! [part8]

//! [part9]
/** input_tensor points to input of a previous network and
    cropROI contains coordinates of output bounding box **/
ov::Tensor input_tensor;
ov::Coordinate begin;
ov::Coordinate end;
//...

/** roi_tensor uses shared memory of input_tensor and describes cropROI
    according to its coordinates **/
ov::Tensor roi_tensor(input_tensor, begin, end);
infer_request2.set_tensor(input_name, roi_tensor);
//! [part9]

//! [part10]
/** Iterate over all input tensors **/
for (auto & item : inputs) {
    /** Create input tensor **/
    ov::Tensor input(item.get_element_type(), item.get_shape());
    infer_request.set_tensor(item.get_any_name(), input);

    /** Fill input tensor with planes. First b channel, then g and r channels **/
//     ...
}
//! [part10]

//! [part11]
infer_request.infer();
//! [part11]


//! [part12]
infer_request.start_async();
infer_request.wait();
//! [part12]

//! [part13]
for (auto &item : outputs) {
    auto output = infer_request.get_tensor(item.get_any_name());
    const float *output_buffer = output.data<const float>();
    /** output_buffer[] - accessing output tensor data **/
}
//! [part13]

return 0;
}

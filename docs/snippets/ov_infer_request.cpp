// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ! [ov:include]
#include <openvino/openvino.hpp>
// ! [ov:include]

int main() {
ov::Core core;
std::shared_ptr<ov::Model> model = core.read_model("model.xml");
ov::CompiledModel compiled_model;

//! [create_infer_request]
auto infer_request = compiled_model.create_infer_request();
//! [create_infer_request]

//! [sync_infer]
infer_request.infer();
//! [sync_infer]

//! [async_infer]
infer_request.start_async();
//! [async_infer]

//! [wait]
infer_request.wait();
//! [wait]

//! [wait_for]
infer_request.wait_for(std::chrono::milliseconds(10));
//! [wait_for]

//! [set_callback]
infer_request.set_callback([&](std::exception_ptr ex_ptr) { 
    if (!ex_ptr) {
        // all done. Output data can be processed.
        // You can fill the input data and run inference one more time:
        infer_request.start_async();
    } else {
        // Something wrong, you can analyze exception_ptr
    }
});
//! [set_callback]

//! [cancel]
infer_request.cancel();
//! [cancel]

{
//! [get_set_one_tensor]
auto input_tensor = infer_request.get_input_tensor();
auto output_tensor = infer_request.get_output_tensor();
//! [get_set_one_tensor]
}

{
//! [get_set_index_tensor]
auto input_tensor = infer_request.get_input_tensor(0);
auto output_tensor = infer_request.get_output_tensor(1);
//! [get_set_index_tensor]
}

//! [get_set_tensor]
auto tensor1 = infer_request.get_tensor("tensor_name1");
ov::Tensor tensor2;
infer_request.set_tensor("tensor_name2", tensor2);
//! [get_set_tensor]

{
//! [get_set_tensor_by_port]
auto input_port = model->input(0);
auto output_port = model->output("tensor_name");
ov::Tensor input_tensor;
infer_request.set_tensor(input_port, input_tensor);
auto output_tensor = infer_request.get_tensor(output_port);
//! [get_set_tensor_by_port]
}

auto infer_request1 = compiled_model.create_infer_request();
auto infer_request2 = compiled_model.create_infer_request();

//! [cascade_models]
auto output = infer_request1.get_output_tensor(0);
infer_request2.set_input_tensor(0, output);
//! [cascade_models]

//! [roi_tensor]
/** input_tensor points to input of a previous network and
    cropROI contains coordinates of output bounding box **/
ov::Tensor input_tensor(ov::element::f32, ov::Shape({1, 3, 20, 20}));
ov::Coordinate begin({0, 0, 0, 0});
ov::Coordinate end({1, 2, 3, 3});
//...

/** roi_tensor uses shared memory of input_tensor and describes cropROI
    according to its coordinates **/
ov::Tensor roi_tensor(input_tensor, begin, end);
infer_request2.set_tensor("input_name", roi_tensor);
//! [roi_tensor]

{
//! [remote_tensor]
ov::RemoteContext context = core.get_default_context("GPU");
auto input_port = compiled_model.input("tensor_name");
ov::RemoteTensor remote_tensor = context.create_tensor(input_port.get_element_type(), input_port.get_shape());
infer_request.set_tensor(input_port, remote_tensor);
//! [remote_tensor]
}

return 0;
}

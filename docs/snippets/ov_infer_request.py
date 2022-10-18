# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino.runtime as ov
#! [import]

core = ov.Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "AUTO")

#! [create_infer_request]
infer_request = compiled_model.create_infer_request()
#! [create_infer_request]

#! [sync_infer]
infer_request.infer()
#! [sync_infer]

#! [async_infer]
infer_request.start_async()
#! [async_infer]

#! [wait]
infer_request.wait()
#! [wait]

#! [wait_for]
infer_request.wait_for(10)
#! [wait_for]

#! [set_callback]
def callback(request, userdata):
    request.start_async()

infer_request.set_callback(callback)
#! [set_callback]

#! [cancel]
infer_request.cancel()
#! [cancel]

#! [get_set_one_tensor]
input_tensor = infer_request.get_input_tensor()
output_tensor = infer_request.get_output_tensor()
#! [get_set_one_tensor]

#! [get_set_index_tensor]
input_tensor = infer_request.get_input_tensor(0)
output_tensor = infer_request.get_output_tensor(1)
#! [get_set_index_tensor]

#! [get_set_name_tensor]
input_tensor = infer_request.get_tensor("input_name")
output_tensor = infer_request.get_tensor("output_name")
#! [get_set_name_tensor]

#! [get_set_tensor]
tensor1 = infer_request.get_tensor("tensor_name1")
tensor2 = ov.Tensor()
infer_request.set_tensor("tensor_name2", tensor2)
#! [get_set_tensor]

#! [get_set_tensor_by_port]
input_port = model.input(0)
output_port = model.input("tensor_name")
input_tensor = ov.Tensor()
infer_request.set_tensor(input_port, input_tensor)
output_tensor = infer_request.get_tensor(output_port)
#! [get_set_tensor_by_port]

infer_request1 = compiled_model.create_infer_request()
infer_request2 = compiled_model.create_infer_request()

#! [cascade_models]
output = infer_request1.get_output_tensor(0)
infer_request2.set_input_tensor(0, output)
#! [cascade_models]

#! [roi_tensor]
# input_tensor points to input of a previous network and
# cropROI contains coordinates of output bounding box **/
input_tensor = ov.Tensor(type=ov.Type.f32, shape=ov.Shape([1, 3, 20, 20]))
begin = [0, 0, 0, 0]
end = [1, 2, 3, 3]
# ...

# roi_tensor uses shared memory of input_tensor and describes cropROI
# according to its coordinates **/
roi_tensor = ov.Tensor(input_tensor, begin, end)
infer_request2.set_tensor("input_name", roi_tensor)
#! [roi_tensor]

#! [remote_tensor]
# NOT SUPPORTED
#! [remote_tensor]

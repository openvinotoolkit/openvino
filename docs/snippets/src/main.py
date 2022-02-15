# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino.runtime as ov
#! [import]

input_name = 'in_name'
output_name = 'out_name'

#! [part0]
core = ov.Core()
model = ov.Model()
compiled_model = ov.CompiledModel()
#! [part0]

#! [part2]
model = ov.Model()
inputs = model.inputs
outputs = model.outputs
#! [part2]

#! [part4_1]
compiled_model = core.compile_model("model.xml", "CPU")
#! [part4_1]
#! [part4_2]
compiled_model = core.compile_model("model.onnx", "CPU")
#! [part4_2]
#! [part4_3]
compiled_model = core.compile_model("model.pdmodel", "CPU")
#! [part4_3]
#! [part4_4]
compiled_model = core.compile_model(model, "CPU")
#! [part4_4]

#! [part5]
# Optional config. E.g. this enables profiling of performance counters.
config = {} # TODO: enable perf counters
compiled_model = core.compile_model(model, "CPU", config)
#! [part5]

#! [part6]
infer_request = compiled_model.create_infer_request()
#! [part6]

infer_request1 = compiled_model.create_infer_request()
infer_request2 = compiled_model.create_infer_request()

#! [part7]
# Iterate over all input tensors
for item in inputs:
    # Get input tensor
    input_tensor = infer_request.get_tensor(item.get_any_name())
    # Fill input tensor
    # ...
#! [part7]

#! [part8]
output_tensor = infer_request1.get_tensor(output_name)
infer_request2.set_tensor(input_name, output_tensor)
#! [part8]

#! [part9]
# input_tensor points to input of a previous network and
# cropROI contains coordinates of output bounding box **/
input_tensor = ov.Tensor()
begin = [0, 0, 0, 0]
end = [1, 2, 3, 3]
# ...

# roi_tensor uses shared memory of input_tensor and describes cropROI
# according to its coordinates **/
roi_tensor = ov.Tensor(input_tensor, begin, end)
infer_request2.set_tensor(input_name, roi_tensor)
#! [part9]

#! [part10]
arr = np.array([1, 2, 3, 4])
input_tensor = ov.Tensor(array=arr, shared_memory=True)
infer_request.set_tensor(input_name, roi_tensor)
#! [part10]

#! [part11]
infer_request.infer()
#! [part11]

#! [part12]
infer_request.start_async()
infer_request.wait()
#! [part12]

#! [part13]
for item in outputs:
    output = infer_request.get_tensor(item.get_any_name())
    output_buffer = output.data
    # output_buffer[] - accessing output tensor data
#! [part13]

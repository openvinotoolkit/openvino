# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino as ov
#! [import]

#! [part1]
core = ov.Core()
#! [part1]

#! [part2_1]
compiled_model = core.compile_model("model.xml", "AUTO")
#! [part2_1]
#! [part2_2]
compiled_model = core.compile_model("model.onnx", "AUTO")
#! [part2_2]
#! [part2_3]
compiled_model = core.compile_model("model.pdmodel", "AUTO")
#! [part2_3]
#! [part2_4]
compiled_model = core.compile_model("model.pb", "AUTO")
#! [part2_4]
#! [part2_5]
compiled_model = core.compile_model("model.tflite", "AUTO")
#! [part2_5]
#! [part2_6]
def create_model():
    # This example shows how to create ov::Function
    #
    # To construct a model, please follow
    # https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application/model-representation.html
    data = ov.opset8.parameter([3, 1, 2], ov.Type.f32)
    res = ov.opset8.result(data)
    return ov.Model([res], [data], "model")

model = create_model()
compiled_model = core.compile_model(model, "AUTO")
#! [part2_6]

#! [part3]
infer_request = compiled_model.create_infer_request()
#! [part3]

memory = np.array([1, 2, 3, 4])
#! [part4]
# Create tensor from external memory
input_tensor = ov.Tensor(array=memory, shared_memory=True)
# Set input tensor for model with one input
infer_request.set_input_tensor(input_tensor)
#! [part4]

#! [part5]
infer_request.start_async()
infer_request.wait()
#! [part5]

#! [part6]
# Get output tensor for model with one output
output = infer_request.get_output_tensor()
output_buffer = output.data
# output_buffer[] - accessing output tensor data
#! [part6]

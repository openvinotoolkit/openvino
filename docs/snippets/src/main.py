# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino.runtime as ov
#! [import]

#! [part1]
core = ov.Core()
model = ov.Model()
compiled_model = ov.CompiledModel()
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
compiled_model = core.compile_model(model, "AUTO")
#! [part2_4]

#! [part3]
infer_request = compiled_model.create_infer_request()
#! [part3]

memory = np.array([1, 2, 3, 4])
#! [part4]
# Get input port for model with one input
input_port = model.input();
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

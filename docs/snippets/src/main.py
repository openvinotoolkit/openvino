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

# ======= FAQ =======

#! [faq:get_set_tensor]
tensor1 = infer_request.get_tensor("tensor_name1")
tensor2 = ov.Tensor()
infer_request.set_tensor("tensor_name2", tensor2)
#! [faq:get_set_tensor]

infer_request1 = compiled_model.create_infer_request()
infer_request2 = compiled_model.create_infer_request()

#! [faq:cascade_models]
output = infer_request1.get_output_tensor(0)
infer_request2.set_input_tensor(0, output)
#! [faq:cascade_models]

#! [faq:roi_tensor]
# input_tensor points to input of a previous network and
# cropROI contains coordinates of output bounding box **/
input_tensor = ov.Tensor()
begin = [0, 0, 0, 0]
end = [1, 2, 3, 3]
# ...

# roi_tensor uses shared memory of input_tensor and describes cropROI
# according to its coordinates **/
roi_tensor = ov.Tensor(input_tensor, begin, end)
infer_request2.set_tensor("input_name", roi_tensor)
#! [faq:roi_tensor]

#! [faq:sync_infer]
infer_request.infer()
#! [faq:sync_infer]

#  
#  #! [part0]
#  core = ov.Core()
#  model = ov.Model()
#  compiled_model = ov.CompiledModel()
#  #! [part0]
#  
#  #! [part2]
#  model = ov.Model()
#  inputs = model.inputs
#  outputs = model.outputs
#  #! [part2]
#  
#  #! [part4_1]
#  compiled_model = core.compile_model("model.xml", "AUTO")
#  #! [part4_1]
#  #! [part4_2]
#  compiled_model = core.compile_model("model.onnx", "AUTO")
#  #! [part4_2]
#  #! [part4_3]
#  compiled_model = core.compile_model("model.pdmodel", "AUTO")
#  #! [part4_3]
#  #! [part4_4]
#  compiled_model = core.compile_model(model, "AUTO")
#  #! [part4_4]
#  
#  #! [part5]
#  # Optional config. E.g. this enables profiling of performance counters.
#  config = {} # TODO: enable perf counters
#  compiled_model = core.compile_model(model, "AUTO", config)
#  #! [part5]
#  
#  #! [part6]
#  infer_request = compiled_model.create_infer_request()
#  #! [part6]
#  
#  infer_request1 = compiled_model.create_infer_request()
#  infer_request2 = compiled_model.create_infer_request()
#  
#  #! [part7]
#  # Iterate over all input tensors
#  for item in inputs:
#      # Get input tensor
#      input_tensor = infer_request.get_tensor(item.get_any_name())
#      # Fill input tensor
#      # ...
#  #! [part7]
#  
#  #! [part8]
#  output_tensor = infer_request1.get_tensor(output_name)
#  infer_request2.set_tensor(input_name, output_tensor)
#  #! [part8]
#  
#  #! [part9]
#  # input_tensor points to input of a previous network and
#  # cropROI contains coordinates of output bounding box **/
#  input_tensor = ov.Tensor()
#  begin = [0, 0, 0, 0]
#  end = [1, 2, 3, 3]
#  # ...
#  
#  # roi_tensor uses shared memory of input_tensor and describes cropROI
#  # according to its coordinates **/
#  roi_tensor = ov.Tensor(input_tensor, begin, end)
#  infer_request2.set_tensor(input_name, roi_tensor)
#  #! [part9]
#  
#  #! [part10]
#  arr = np.array([1, 2, 3, 4])
#  input_tensor = ov.Tensor(array=arr, shared_memory=True)
#  infer_request.set_tensor(input_name, roi_tensor)
#  #! [part10]
#  
#  #! [faq:sync_infer]
#  infer_request.infer()
#  #! [faq:sync_infer]
#  
#  #! [part12]
#  infer_request.start_async()
#  infer_request.wait()
#  #! [part12]
#  
#  #! [part13]
#  for item in outputs:
#      output = infer_request.get_tensor(item.get_any_name())
#      output_buffer = output.data
#      # output_buffer[] - accessing output tensor data
#  #! [part13]

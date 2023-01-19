# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino.runtime as ov
#! [import]

#! [reshape_undefined]
core = ov.Core()
model = core.read_model("model.xml")

# Set one static dimension (= 1) and another dynamic dimension (= Dimension())
model.reshape([1, ov.Dimension()])

# The same as above
model.reshape([1, -1])

# The same as above
model.reshape("1, ?")

# Or set both dimensions as dynamic if both are going to be changed dynamically
model.reshape([ov.Dimension(), ov.Dimension()])

# The same as above
model.reshape([-1, -1])

# The same as above
model.reshape("?, ?")
#! [reshape_undefined]

#! [reshape_bounds]
# Example 1 - set first dimension as dynamic (no bounds) and third and fourth dimensions to range of 112..448
model.reshape([-1, 3, (112, 448), (112, 448)])

# Example 2 - Set first dimension to a range of 1..8 and third and fourth dimensions to range of 112..448
model.reshape([(1, 8), 3, (112, 448), (112, 448)])
#! [reshape_bounds]

model = core.read_model("model.xml")

#! [print_dynamic]
# Print output partial shape
print(model.output().partial_shape)

# Print input partial shape
print(model.input().partial_shape)
#! [print_dynamic]

#! [detect_dynamic]
model = core.read_model("model.xml")

if model.input(0).partial_shape.is_dynamic():
    # input is dynamic
    pass

if model.output(0).partial_shape.is_dynamic():
    # output is dynamic
    pass

if model.output(0).partial_shape[1].is_dynamic():
    # 1-st dimension of output is dynamic
    pass
#! [detect_dynamic]

executable = core.compile_model(model)
infer_request = executable.create_infer_request()

#! [set_input_tensor]
# For first inference call, prepare an input tensor with 1x128 shape and run inference request
Input_data1 = np.ones(shape=[1,128])
infer_request.infer([input_data1])

# Get resulting outputs
Output_tensor1 = infer_request.get_output_tensor()
Output_data1 = output_tensor.data[:]

# For second inference call, prepare a 1x200 input tensor and run inference request
Input_data2 = np.ones(shape=[1,200])
infer_request.infer([input_data2])

# Get resulting outputs
Output_tensor2 = infer_request.get_output_tensor()
Output_data2 = output_tensor.data[:]
#! [set_input_tensor]

infer_request = executable.create_infer_request()

#! [get_input_tensor]
# Get the tensor, shape is not initialized
input_tensor = infer_request.get_input_tensor()

# Set shape is required
input_tensor.shape = [1, 128]
# ... write values to input_tensor

infer_request.infer()
output_tensor = infer_request.get_output_tensor()
data1 = output_tensor.data[:]

# The second inference call, repeat steps:

# Set a new shape, may reallocate tensor memory
input_tensor.shape = [1, 200]
# ... write values to input_tensor

infer_request.infer()
data2 = output_tensor.data[:]
#! [get_input_tensor]

# Copyright (C) 2018-2022 Intel Corporation
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
# Both dimensions are dynamic, first has a size within 1..10 and the second has a size within 8..512
model.reshape([ov.Dimension(1, 10), ov.Dimension(8, 512)])

# The same as above
model.reshape([(1, 10), (8, 512)])

# The same as above
model.reshape("1..10, 8..512")

# Both dimensions are dynamic, first doesn't have bounds, the second is in the range of 8..512
model.reshape([-1, (8, 512)])
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
# The first inference call

# Create tensor compatible to the model input
# Shape {1, 128} is compatible with any reshape statements made in previous examples
input_tensor1 = ov.Tensor(model.input().element_type, [1, 128])
# ... write values to input_tensor_1

# Set the tensor as an input for the infer request
infer_request.set_input_tensor(input_tensor1)

# Do the inference
infer_request.infer()

# Or pass a tensor in infer to set the tensor as a model input and make the inference
infer_request.infer([input_tensor1])

# Or pass the numpy array to set inputs of the infer request
input_data = np.ones(shape=[1, 128])
infer_request.infer([input_data])

# Retrieve a tensor representing the output data
output_tensor = infer_request.get_output_tensor()

# Copy data from tensor to numpy array
data1 = output_tensor.data[:]

# The second inference call, repeat steps:

# Create another tensor (if the previous one cannot be utilized)
# Notice, the shape is different from input_tensor_1
input_tensor2 = ov.Tensor(model.input().element_type, [1, 200])
# ... write values to input_tensor_2

infer_request.infer([input_tensor2])

# No need to call infer_request.get_output_tensor() again
# output_tensor queried after the first inference call above is valid here.
# But it may not be true for the memory underneath as shape changed, so re-take an output data:
data2 = output_tensor.data[:]
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

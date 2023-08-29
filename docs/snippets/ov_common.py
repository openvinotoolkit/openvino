# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


import numpy as np
from utils import get_image, get_path_to_extension_library, get_path_to_model

#! [ov_api_2_0:create_core]
import openvino as ov
core = ov.Core()
#! [ov_api_2_0:create_core]

model_path = get_path_to_model()

#! [ov_api_2_0:read_model]
model = core.read_model(model_path)
#! [ov_api_2_0:read_model]

#! [ov_api_2_0:compile_model]
compiled_model = core.compile_model(model, "CPU")
#! [ov_api_2_0:compile_model]

#! [ov_api_2_0:create_infer_request]
infer_request = compiled_model.create_infer_request()
#! [ov_api_2_0:create_infer_request]

#! [ov_api_2_0:get_input_tensor_aligned]
# Get input tensor by index
input_tensor1 = infer_request.get_input_tensor(0)
# Element types, names and layouts are aligned with framework
assert input_tensor1.data.dtype == np.int64
# Fill the first data ...

# Get input tensor by tensor name
input_tensor2 = infer_request.get_tensor("input")
assert input_tensor2.data.dtype == np.int64
# Fill the second data ...
#! [ov_api_2_0:get_input_tensor_aligned]

#! [ov_api_2_0:get_input_tensor_v10]
# Get input tensor by index
input_tensor1 = infer_request.get_input_tensor(0)
# IR v10 works with converted precisions (i64 -> i32)
assert input_tensor1.data.dtype == np.int32
# Fill the first data ...

# Get input tensor by tensor name
input_tensor2 = infer_request.get_tensor("input")
# IR v10 works with converted precisions (i64 -> i32)
assert input_tensor2.data.dtype == np.int32
# Fill the second data ..
#! [ov_api_2_0:get_input_tensor_v10]

#! [ov_api_2_0:inference]
results = infer_request.infer()
#! [ov_api_2_0:inference]

input_data = get_image()

def process_results(results, frame_id):
    pass

#! [ov_api_2_0:start_async_and_wait]
# Start async inference on a single infer request
infer_request.start_async()
# Wait for 1 milisecond
infer_request.wait_for(1)
# Wait for inference completion
infer_request.wait()

# Demonstrates async pipeline using AsyncInferQueue

results = []

def callback(request, frame_id):
    # Copy the data from output tensors to numpy array and process it
    results_copy = {output: data[:] for output, data in request.results.items()}
    results.append(process_results(results_copy, frame_id))

# Create AsyncInferQueue with 4 infer requests
infer_queue = ov.AsyncInferQueue(compiled_model, jobs=4)
# Set callback for each infer request in the queue
infer_queue.set_callback(callback)

total_frames = 100
for i in range(total_frames):
    # Wait for at least one available infer request and start asynchronous inference
    infer_queue.start_async(input_data, userdata=i)
# Wait for all requests to complete
infer_queue.wait_all()
#! [ov_api_2_0:start_async_and_wait]

#! [ov_api_2_0:get_output_tensor_aligned]
# Model has only one output
output_tensor = infer_request.get_output_tensor()
# Element types, names and layouts are aligned with framework
assert output_tensor.data.dtype == np.int64
# process output data ...
#! [ov_api_2_0:get_output_tensor_aligned]

#! [ov_api_2_0:get_output_tensor_v10]
# Model has only one output
output_tensor = infer_request.get_output_tensor()
# IR v10 works with converted precisions (i64 -> i32)
assert output_tensor.data.dtype == np.int32
# process output data ...
#! [ov_api_2_0:get_output_tensor_v10]

path_to_extension_library = get_path_to_extension_library()

#! [ov_api_2_0:load_old_extension]
core.add_extension(path_to_extension_library)
#! [ov_api_2_0:load_old_extension]

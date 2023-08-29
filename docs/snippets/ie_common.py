# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from utils import get_path_to_model, get_image, get_path_to_extension_library

#! [ie:create_core]
import numpy as np
import openvino.inference_engine as ie
core = ie.IECore()
#! [ie:create_core]

model_path = get_path_to_model(True)

#! [ie:read_model]
network = core.read_network(model_path)
#! [ie:read_model]

#! [ie:compile_model]
# Load network to the device and create infer requests
exec_network = core.load_network(network, "CPU", num_requests=4)
#! [ie:compile_model]

#! [ie:create_infer_request]
# Done in the previous step
#! [ie:create_infer_request]

#! [ie:get_input_tensor]
infer_request = exec_network.requests[0]
# Get input blobs mapped to input layers names
input_blobs = infer_request.input_blobs
data = input_blobs["data"].buffer
# Original I64 precision was converted to I32
assert data.dtype == np.int32
# Fill the first blob ...
#! [ie:get_input_tensor]

#! [ie:inference]
results = infer_request.infer()
#! [ie:inference]

input_data = get_image()

def process_results(results, frame_id):
    pass

#! [ie:start_async_and_wait]
# Start async inference on a single infer request
infer_request.async_infer()
# Wait for 1 milisecond
infer_request.wait(1)
# Wait for inference completion
infer_request.wait()

# Demonstrates async pipeline using ExecutableNetwork

results = []

# Callback to process inference results
def callback(output_blobs, _):
    # Copy the data from output blobs to numpy array
    results_copy = {out_name: out_blob.buffer[:] for out_name, out_blob in output_blobs.items()}
    results.append(process_results(results_copy))

# Setting callback for each infer requests
for infer_request in exec_network.requests:
    infer_request.set_completion_callback(callback, py_data=infer_request.output_blobs)

# Async pipline is managed by ExecutableNetwork
total_frames = 100
for _ in range(total_frames):
    # Wait for at least one free request
    exec_network.wait(num_requests=1)
    # Get idle id
    idle_id = exec_network.get_idle_request_id()
    # Start asynchronous inference on idle request
    exec_network.start_async(request_id=idle_id, inputs={"data": input_data})
# Wait for all requests to complete
exec_network.wait()
#! [ie:start_async_and_wait]

#! [ie:get_output_tensor]
# Get output blobs mapped to output layers names
output_blobs = infer_request.output_blobs
data = output_blobs["relu"].buffer
# Original I64 precision was converted to I32
assert data.dtype == np.int32
# Process output data
#! [ie:get_output_tensor]

path_to_extension_library = get_path_to_extension_library()
#! [ie:load_old_extension]
core.add_extension(path_to_extension_library, "CPU")
#! [ie:load_old_extension]

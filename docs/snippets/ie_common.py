# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#


#! [ie:create_core]
from urllib import request
import openvino.inference_engine as ie
core = ie.IECore()
#! [ie:create_core]

#! [ie:create_core]
network = core.read_network("model.xml")
#! [ie:create_core]

#! [ie:compile_model]
# Load network to the device and create infer requests
exec_network = core.load_network(network, "CPU", num_requests=4)
#! [ie:compile_model]

#! [ie:create_infer_request]
# Done in the previous step
#! [ie:create_infer_request]

input_data = {}

#! [ie:get_input_tensor]
request = exec_network.requests[0]
# Get inpub blobs mapped to input layer names
input_blobs = request.input_blobs
# Copy input data in input blobs
for input_name in input_blobs:
    input_blobs[input_name].buffer[:] = input_data[input_name]

# Or just pass the data in infer() to fill input blobs and run inference
request.infer(input_data)
#! [ie:get_input_tensor]

#! [ie:inference]
request.infer()
#! [ie:inference]

from copy import deepcopy

#! [ie:start_async_and_wait]
# Start async inference on a single infer request
request.async_infer()
# Wait for 1 milisecond
request.wait(1)
# Wait for inference completion
request.wait()

# Demonstrates async pipeline using ExecutableNetwork
results = []

# Callback to process inference results
def callback(output_blobs, status_code):
    # copy output blobs data
    outputs_copy = deepcopy(output_blobs)
    results.append(outputs_copy)

# Setting callback for each infer requests
for request in exec_network.requests:
    request.set_completion_callback(callback, py_data=request.output_blobs)

# Async pipline is managed by ExecutableNetwork
total_frames = 100
for _ in range(total_frames):
    # Wait for at least one free request
    exec_network.wait(num_request=1)
    # Get idle id
    idle_id = exec_network.get_idle_request_id()
    # Start asynchronous inference on idle request
    exec_network.start_async(request_id=idle_id, inputs=next(input_data))
# Wait for the rest requests to complete
exec_network.wait()
#! [ie:start_async_and_wait]

#! [ie:get_output_tensor]
# Get inference results mapped to output layers names
results = request.infer(input_data)
# Acessing output blobs directly
output_blobs = request.output_blobs
#! [ie:get_output_tensor]

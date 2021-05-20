# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
First example.

User wants to use OpenVINO API to infer one picture using
synchronous Infer Request.
"""

import numpy as np

from openvino.inference_engine import IECore
from openvino.inference_engine import TensorDesc
from openvino.inference_engine import Blob

import helpers

# Generate random image
img = helpers.generate_random_images(num=1)[0]

# Read and Load of network
ie = IECore()
ie_network = ie.read_network(
    helpers.get_example_model_path(),
    helpers.get_example_weights_path())
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})

# Infer directly
result_executable_network = executable_network.infer({'data': img})

# Prepare request
request = executable_network.create_infer_request()

# Create blob
tensor_desc = TensorDesc('FP32',
                         [1, 3, img.shape[2], img.shape[3]],
                         'NCHW')
img_blob = Blob(tensor_desc, img)

# Set it by using dedicated function
request.set_blob('data', img_blob)
assert np.allclose(img, request.get_blob('data').buffer)
# Or more versitile function
request.set_input({'data': img_blob})
assert np.allclose(img, request.get_blob('data').buffer)

request.set_input({'data': np.ascontiguousarray(img)})
assert np.allclose(img, request.get_blob('data').buffer)
# And infer
result_infer_request = request.infer()

# Or just use it directly
result_infer_request = request.infer({'data': img})

# Print results using output names from network output_info
for key in executable_network.output_info:
    print("Executable Network result: ")
    print(result_executable_network[key])
    print("Infer request result: ")
    print(result_infer_request[key])
    assert np.allclose(result_infer_request[key],
                       result_executable_network[key])

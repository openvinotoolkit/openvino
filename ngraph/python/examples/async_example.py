# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""
Second example.

User wants to use OpenVINO API to infer one picture using
asynchronous Infer Request.
"""

import numpy as np

from openvino.inference_engine import IECore
from openvino.inference_engine import StatusCode

import helpers


def get_reference(executable_network, image):
    """Get reference outputs using synchronous API."""
    return executable_network.infer({'data': img})


# Read images from a folder
img = helpers.generate_random_images(num=1)[0]

# Read and Load of network
ie = IECore()
ie_network = ie.read_network(
    helpers.get_example_model_path(),
    helpers.get_example_weights_path())
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})

ref_result = get_reference(executable_network, img)

# Create InferRequest
request = executable_network.create_infer_request()


# Create callback function
def say_hi(request, status, userdata):
    """User-defined callback function."""
    print('This is your Infer Request named',
          userdata,
          ', I am done! Returning',
          status)
    if status != StatusCode.OK:
        raise RuntimeError('Infer Request returns with ', status)
    print('Results in callback:\n', request.get_result('fc_out'), sep='')


# Set callback on request
request.set_completion_callback(say_hi, 'My First Async Infer')

print('Starting async infer...')
request.async_infer({'data': img})

print('You can do something here!')

# Wait for Infer Request to finish
status = request.wait()

if status == StatusCode.OK:
    print('Finished asynchronous infer!')
    for key in executable_network.output_info:
        print(request.get_blob(key).buffer)
        print(ref_result[key])
        assert np.allclose(request.get_blob(key).buffer,
                           ref_result[key])
else:
    raise RuntimeError('Infer Request failed to finish!')

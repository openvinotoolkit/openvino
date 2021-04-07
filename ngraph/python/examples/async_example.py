"""
First example.

User wants to use OpenVINO API to infer one picture using
synchronous Infer Request.
"""

import numpy as np
import time

from openvino.inference_engine import IECore
from openvino.inference_engine import TensorDesc
from openvino.inference_engine import Blob
from openvino.inference_engine import StatusCode

from helpers import get_images


def get_reference(executable_network, image):
    """Get reference outputs using synchronous API."""
    return executable_network.infer({'data': img})


# Read images from a folder
images = get_images()

# Read and Load of network
ie = IECore()
ie_network = ie.read_network(
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.xml',
    '/home/jiwaszki/testdata/models/test_model/test_model_fp32.bin')
executable_network = ie.load_network(network=ie_network,
                                     device_name='CPU',
                                     config={})

img = images[0]
ref_result = get_reference(executable_network, img)

# Create InferRequest
request = executable_network.create_infer_request()

num_of_runs = 1

times = []


# Create callback function
def say_hi(request, userdata):
    """User-defined callback function."""
    print("Hi this is your Infer Request, I'm done!", userdata)
    print(request.output_blobs['fc_out'].buffer.copy())
    global times
    times += [(time.time() - userdata) * 1000]


# Set callback on our request
# Async infer
print('Starting async infer...')

request.infer()

for i in range(num_of_runs):
    start_time = time.time()
    request.set_completion_callback(say_hi, start_time)
    request.async_infer({'data': img})
    status = request.wait()

print(times)
latency_median = np.median(np.array(times))
print(latency_median)

# print('I can do something here!')
# # # Do some work
# # j = 0
# # for i in range(1000000):
# #     j = i
# # print("j =", j)
# time.sleep(3)

# Wait for Infer Request to finish
status = request.wait()

# print(request.get_perf_counts())

if status == StatusCode.OK:
    print('Finished asynchronous infer!')
    for key in executable_network.output_info:
        print(request.get_blob(key).buffer)
        print(ref_result[key])
        assert np.allclose(request.get_blob(key).buffer,
                           ref_result[key])
else:
    raise RuntimeError('Infer Request failed to finish!')

# TODO: When callback is present everything works
# del request

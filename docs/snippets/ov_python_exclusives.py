# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

#! [auto_compilation]
import openvino.runtime as ov

compiled_model = ov.compile_model("model.xml")
#! [auto_compilation]

#! [properties_example]
core = ov.Core()

input_a = ov.opset8.parameter([8])
res = ov.opset8.absolute(input_a)
model = ov.Model(res, [input_a])
compiled = core.compile_model(model, "CPU")

print(model.inputs)
print(model.outputs)

print(compiled.inputs)
print(compiled.outputs)
#! [properties_example]

#! [tensor_basics]
data_float64 = np.ones(shape=(2,8))

tensor = ov.Tensor(data_float64)
assert tensor.element_type == ov.Type.f64

data_int32 = np.ones(shape=(2,8), dtype=np.int32)

tensor = ov.Tensor(data_int32)
assert tensor.element_type == ov.Type.i32
#! [tensor_basics]

#! [tensor_shared_mode]
data_to_share = np.ones(shape=(2,8))

shared_tensor = ov.Tensor(data_to_share, shared_memory=True)

# Editing of the numpy array affects Tensor's data
data_to_share[0][2] = 6.0
assert shared_tensor.data[0][2] == 6.0

# Editing of Tensor's data affects the numpy array
shared_tensor.data[0][2] = 0.6
assert data_to_share[0][2] == 0.6
#! [tensor_shared_mode]

infer_request = compiled.create_infer_request()
data = np.random.randint(-5, 3 + 1, size=(8))

#! [passing_numpy_array]
# Passing inputs data in form of a dictionary
infer_request.infer(inputs={0: data})
# Passing inputs data in form of a list
infer_request.infer(inputs=[data])
#! [passing_numpy_array]

#! [getting_results]
# Get output tensor
results = infer_request.get_output_tensor().data

# Get tensor with CompiledModel's output node
results = infer_request.get_tensor(compiled.outputs[0]).data

# Get all results with special helper property
results = list(infer_request.results.values())
#! [getting_results]

#! [sync_infer]
# Simple call to InferRequest
results = infer_request.infer(inputs={0: data})
# Extra feature: calling CompiledModel directly
results = compiled_model(inputs={0: data})
#! [sync_infer]

#! [asyncinferqueue]
core = ov.Core()

# Simple model that adds two inputs together
input_a = ov.opset8.parameter([8])
input_b = ov.opset8.parameter([8])
res = ov.opset8.add(input_a, input_b)
model = ov.Model(res, [input_a, input_b])
compiled = core.compile_model(model, "CPU")

# Number of InferRequests that AsyncInferQueue holds
jobs = 4
infer_queue = ov.AsyncInferQueue(compiled, jobs)

# Create data
data = [np.array([i] * 8, dtype=np.float32) for i in range(jobs)]

# Run all jobs
for i in range(len(data)):
    infer_queue.start_async({0: data[i], 1: data[i]})
infer_queue.wait_all()
#! [asyncinferqueue]

#! [asyncinferqueue_access]
results = infer_queue[3].get_output_tensor().data
#! [asyncinferqueue_access]

#! [asyncinferqueue_set_callback]
data_done = [False for _ in range(jobs)]

def f(request, userdata):
    print(f"Done! Result: {request.get_output_tensor().data}")
    data_done[userdata] = True

infer_queue.set_callback(f)

for i in range(len(data)):
    infer_queue.start_async({0: data[i], 1: data[i]}, userdata=i)
infer_queue.wait_all()

assert all(data_done)
#! [asyncinferqueue_set_callback]

unt8_data = np.ones([100])

#! [packing_data]
from openvino.helpers import pack_data

packed_buffer = pack_data(unt8_data, ov.Type.u4)
# Create tensor with shape in element types
t = ov.Tensor(packed_buffer, [1, 128], ov.Type.u4)
#! [packing_data]

#! [unpacking]
from openvino.helpers import unpack_data

unpacked_data = unpack_data(t.data, t.element_type, t.shape)
assert np.array_equal(unpacked_data , unt8_data)
#! [unpacking]

#! [releasing_gil]
import openvino.runtime as ov
import cv2 as cv
from threading import Thread

input_data = []

# Processing input data will be done in a separate thread
# while compilation of the model and creation of the infer request
# is going to be executed in the main thread.
def prepare_data(input, image_path):
    image = cv.imread(image_path)
    h, w = list(input.shape)[-2:]
    image = cv.resize(image, (h, w))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    input_data.append(image)

core = ov.Core()
model = core.read_model("model.xml")
# Create thread with prepare_data function as target and start it
thread = Thread(target=prepare_data, args=[model.input(), "path/to/image"])
thread.start()
# The GIL will be released in compile_model.
# It allows a thread above to start the job,
# while main thread is running in the background.
compiled = core.compile_model(model, "GPU")
# After returning from compile_model, the main thread acquires the GIL
# and starts create_infer_request which releases it once again.
request = compiled.create_infer_request()
# Join the thread to make sure the input_data is ready
thread.join()
# running the inference
request.infer(input_data)
#! [releasing_gil]

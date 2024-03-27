# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov

from utils import get_path_to_model, get_image

#! [properties_example]
import openvino.runtime.opset12 as ops

core = ov.Core()

input_a = ops.parameter([8], name="input_a")
res = ops.absolute(input_a)
model = ov.Model(res, [input_a])
compiled = core.compile_model(model, "CPU")
model.outputs[0].tensor.set_names({"result_0"})  # Add name for Output

print(model.inputs)
print(model.outputs)

print(compiled.inputs)
print(compiled.outputs)
#! [properties_example]

#! [auto_compilation]
compiled_model = ov.compile_model(model)
#! [auto_compilation]

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

#! [ov_dict]
results = compiled_model(inputs={0: data})

# Access via string
_ = results["result_0"]
# Access via index
_ = results[0]
# Access via output port
_ = results[compiled_model.outputs[0]]
# Use iterator over keys
_ = results[next(iter(results))]
# Iterate over values
_ = next(iter(results.values()))
#! [ov_dict]

#! [asyncinferqueue]
core = ov.Core()

# Simple model that adds two inputs together
input_a = ops.parameter([8])
input_b = ops.parameter([8])
res = ops.add(input_a, input_b)
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

unt8_data = np.ones([100], dtype=np.uint8)

#! [packing_data]
from openvino.helpers import pack_data

packed_buffer = pack_data(unt8_data, ov.Type.u4)
# Create tensor with shape in element types
t = ov.Tensor(packed_buffer, [100], ov.Type.u4)
#! [packing_data]

#! [unpacking]
from openvino.helpers import unpack_data

unpacked_data = unpack_data(t.data, t.element_type, t.shape)
assert np.array_equal(unpacked_data , unt8_data)
#! [unpacking]

model_path = get_path_to_model()
image = get_image()


#! [releasing_gil]
import openvino as ov
from threading import Thread

input_data = []

# Processing input data will be done in a separate thread
# while compilation of the model and creation of the infer request
# is going to be executed in the main thread.
def prepare_data(input, image):
    shape = list(input.shape)
    resized_img = np.resize(image, shape)
    input_data.append(resized_img)

core = ov.Core()
model = core.read_model(model_path)
# Create thread with prepare_data function as target and start it
thread = Thread(target=prepare_data, args=[model.input(), image])
thread.start()
# The GIL will be released in compile_model.
# It allows a thread above to start the job,
# while main thread is running in the background.
compiled = core.compile_model(model, "CPU")
# After returning from compile_model, the main thread acquires the GIL
# and starts create_infer_request which releases it once again.
request = compiled.create_infer_request()
# Join the thread to make sure the input_data is ready
thread.join()
# running the inference
request.infer(input_data)
#! [releasing_gil]

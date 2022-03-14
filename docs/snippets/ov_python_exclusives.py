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
res = ov.opset8.relu(input_a)
model = ov.Model(res, [input_a])
compiled = core.compile_model(model, "CPU")

print(model.inputs)
print(model.outputs)

print(compiled.inputs)
print(compiled.outputs)
#! [properties_example]

#! [tensor_basics]
data_float64 = np.ones(shape=(2,8))

t = ov.Tensor(data_float64)
assert t.element_type == ov.Type.f64

data_int32 = np.ones(shape=(2,8), dtype=np.int32)

t = ov.Tensor(data_int32)
assert t.element_type == ov.Type.i32
#! [tensor_basics]

#! [tensor_shared_mode]
data_to_share = np.ones(shape=(2,8))

t_shared = ov.Tensor(data_to_share, shared_memory=True)

# Editing of the numpy array affects Tensor's data
data_to_share[0][2] = 6.0
assert t_shared.data[0][2] == 6.0

# Editing of Tensor's data affects the numpy array
t_shared.data[0][2] = 0.6
assert data_to_share[0][2] == 0.6
#! [tensor_shared_mode]

#! [tensor_slice_mode]
data_to_share = np.ones(shape=(2,8))

# Specify slice of memory and the shape
t_shared = ov.Tensor(data_to_share[1][:] , shape=ov.Shape([8]))

# Editing of the numpy array affects Tensor's data
data_to_share[1][:] = 2
assert np.array_equal(t_shared.data, data_to_share[1][:])
#! [tensor_slice_mode]

infer_request = compiled_model.create_infer_request()
data = np.ones(shape=(2,8))

#! [passing_numpy_array]
# Passing inputs data in form of a dictionary
infer_request.infer(inputs={0: data})
# Passing inputs data in form of a list
infer_request.infer(inputs=[data])
#! [passing_numpy_array]

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

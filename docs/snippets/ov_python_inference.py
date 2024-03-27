# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import openvino.runtime.opset12 as ops

INPUT_SIZE = 1_000_000  # Use bigger values if necessary, i.e.: 300_000_000

input_0 = ops.parameter([INPUT_SIZE], name="input_0")
input_1 = ops.parameter([INPUT_SIZE], name="input_1")
add_inputs = ops.add(input_0, input_1)
res = ops.reduce_sum(add_inputs, reduction_axes=0, name="reduced")
model = ov.Model(res, [input_0, input_1], name="my_model")
model.outputs[0].tensor.set_names({"reduced_result"})  # Add name for Output

core = ov.Core()
compiled_model = core.compile_model(model, device_name="CPU")

data_0 = np.array([0.1] * INPUT_SIZE, dtype=np.float32)
data_1 = np.array([-0.1] * INPUT_SIZE, dtype=np.float32)

data_2 = np.array([0.2] * INPUT_SIZE, dtype=np.float32)
data_3 = np.array([-0.2] * INPUT_SIZE, dtype=np.float32)

#! [direct_inference]
# Calling CompiledModel creates and saves InferRequest object
results_0 = compiled_model({"input_0": data_0, "input_1": data_1})
# Second call reuses previously created InferRequest object
results_1 = compiled_model({"input_0": data_2, "input_1": data_3})
#! [direct_inference]

request = compiled_model.create_infer_request()

#! [shared_memory_inference]
# Data can be shared only on inputs
_ = compiled_model({"input_0": data_0, "input_1": data_1}, share_inputs=True)
_ = request.infer({"input_0": data_0, "input_1": data_1}, share_inputs=True)
# Data can be shared only on outputs
_ = request.infer({"input_0": data_0, "input_1": data_1}, share_outputs=True)
# Or both flags can be combined to achieve desired behavior
_ = compiled_model({"input_0": data_0, "input_1": data_1}, share_inputs=False, share_outputs=True)
#! [shared_memory_inference]

time_in_sec = 2.0

#! [hiding_latency]
import time

# Long running function
def run(time_in_sec):
    time.sleep(time_in_sec)

# No latency hiding
results = request.infer({"input_0": data_0, "input_1": data_1})[0]
run(time_in_sec)

# Hiding latency
request.start_async({"input_0": data_0, "input_1": data_1})
run(time_in_sec)
request.wait()
results = request.get_output_tensor(0).data  # Gather data from InferRequest
#! [hiding_latency]

#! [no_return_inference]
# Standard approach
results = request.infer({"input_0": data_0, "input_1": data_1})[0]

# "Postponed Return" approach
request.start_async({"input_0": data_0, "input_1": data_1})
request.wait()
results = request.get_output_tensor(0).data  # Gather data "on demand" from InferRequest
#! [no_return_inference]


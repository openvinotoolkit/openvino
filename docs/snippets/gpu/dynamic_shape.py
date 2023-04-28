# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.runtime as ov

#! [dynamic_shape]
core = ov.Core()

model = core.read_model("model.xml")
model.reshape([-1, -1])

# compile model and create infer request
compiled_model = core.compile_model(model, "GPU")
infer_request = compiled_model.create_infer_request()

# create input tensor with specific shape
input_tensor = ov.Tensor(model.input().element_type, [2, 224])

# ...

infer_request.infer([input_tensor])

#! [dynamic_shape]

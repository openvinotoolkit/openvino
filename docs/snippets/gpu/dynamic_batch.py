# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino.runtime as ov

#! [dynamic_batch]
core = ov.Core()

C = 3
H = 224
W = 224

model = core.read_model("model.xml")
model.reshape([(1, 10), C, H, W])

# compile model and create infer request
compiled_model = core.compile_model(model, "GPU")
infer_request = compiled_model.create_infer_request()

# create input tensor with specific batch size
input_tensor = ov.Tensor(model.input().element_type, [2, C, H, W])

# ...

infer_request.infer([input_tensor])

#! [dynamic_batch]

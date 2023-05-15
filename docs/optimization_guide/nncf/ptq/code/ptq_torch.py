# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import torch

calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [quantization]
model = ... # torch.nn.Module object

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]

#! [inference]
import openvino.runtime as ov
from openvino.tools.mo import convert_model

input_fp32 = ... # FP32 model input

# convert PyTorch model to OpenVINO model
ov_quantized_model = convert_model(quantized_model, example_input=input_fp32)

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(ov_quantized_model)

res = model_int8(input_fp32)
#! [inference]
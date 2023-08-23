# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import torch

calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return images.numpy()

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [quantization]
model = ... # openvino.runtime.Model object

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]

#! [inference]
import openvino as ov

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(quantized_model)

input_fp32 = ... # FP32 model input
res = model_int8(input_fp32)

# save the model
ov.serialize(quantized_model, "quantized_model.xml")
#! [inference]

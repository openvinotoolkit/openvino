# Copyright (C) 2018-2025 Intel Corporation
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
import openvino as ov
model = ov.Core().read_model("model_path")

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]

#! [inference]
# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(quantized_model)

input_fp32 = ... # FP32 model input
res = model_int8(input_fp32)

# save the model
ov.save_model(quantized_model, "quantized_model.xml")
#! [inference]

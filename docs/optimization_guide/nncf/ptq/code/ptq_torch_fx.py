# Copyright (C) 2018-2025 Intel Corporation
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
import torchvision
from nncf.torch import disable_patching

input_fp32 = torch.ones((1, 3, 224, 224)) # FP32 model input
model = torchvision.models.resnet50(pretrained=True)

with disable_patching():
    exported_model = torch.export.export_for_training(model, args=(input_fp32,)).module()
    quantized_model = nncf.quantize(exported_model, calibration_dataset)
#! [quantization]

#! [inference]
import openvino.torch

input_fp32 = ... # FP32 model input

# compile quantized model using torch.compile API
with disable_patching():
    compiled_model_int8 = torch.compile(quantized_model, backend="openvino")
    # OpenVINO backend compiles the model during the first call,
    # so the first call is expected to be slower than the following calls
    res = compiled_model_int8(input_fp32)

    # save the model
    exported_program = torch.export.export(quantized_model, args=(input_fp32,))
    torch.export.save(exported_program, 'exported_program.pt2')
#! [inference]

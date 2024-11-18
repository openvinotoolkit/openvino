# Copyright (C) 2018-2024 Intel Corporation
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
import openvino.torch
model = torchvision.models.resnet50(pretrained=True)

input_fp32 = ... # FP32 model input

with nncf.torch.disable_patching():
    exported_model = torch.export.export_for_training(model, args=(input_fp32,)).module()
    quantized_model = nncf.quantize(exported_model, calibration_dataset)
#! [quantization]

#! [inference]
import openvino as ov

input_fp32 = ... # FP32 model input

# compile quantized model using torch.compile API
with nncf.torch.disable_patching():
    compiled_model_int8 = torch.compile(quantized_model, backend="openvino")
    # First call compiles an OpenVINO model underneath, so it could take longer
    # than original model call.
    res = compiled_model_int8(input_fp32)
    print(res) # torch.tensor(...)
...


# convert exported Torch model to OpenVINO model
with nncf.torch.disable_patching():
    exported_quantized_model = torch.export.export(quantized_model, args=(input_fp32,))
    ov_quantized_model = ov.convert_model(exported_quantized_model, example_input=input_fp32)

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(ov_quantized_model)

res = model_int8(input_fp32)
print(res) # torch.tensor(...)


# save the model
ov.save_model(ov_quantized_model, "quantized_model.xml")
#! [inference]

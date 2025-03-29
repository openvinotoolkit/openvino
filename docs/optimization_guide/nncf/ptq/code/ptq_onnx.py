# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import torch

calibration_loader = torch.utils.data.DataLoader(...)

def transform_fn(data_item):
    images, _ = data_item
    return {input_name: images.numpy()} # input_name should be taken from the model, 
                                        # e.g. model.graph.input[0].name

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [quantization]
import onnx
model = onnx.load("model_path")

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]

#! [inference]
import openvino as ov

# use a temporary file to convert ONNX model to OpenVINO model
quantized_model_path = "quantized_model.onnx"
onnx.save(quantized_model, quantized_model_path)

ov_quantized_model = ov.convert_model(quantized_model_path)

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(ov_quantized_model)

input_fp32 = ... # FP32 model input
res = model_int8(input_fp32)

# save the model
ov.save_model(ov_quantized_model, "quantized_model.xml")
#! [inference]

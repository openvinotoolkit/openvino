# Copyright (C) 2018-2024 Intel Corporation
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
validation_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [validation]
import numpy as np
import torch
from sklearn.metrics import accuracy_score

import onnx
import onnxruntime


def validate(model: onnx.ModelProto,
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    input_name = model.graph.input[0].name
    serialized_model = model.SerializeToString()
    session = onnxruntime.InferenceSession(serialized_model, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]

    for images, target in validation_loader:
        pred = session.run(output_names, input_feed={input_name: images.numpy()})[0]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)
#! [validation]

#! [quantization]
import onnx

model = onnx.load("model_path")

quantized_model = nncf.quantize_with_accuracy_control(
    model,
    calibration_dataset=calibration_dataset,
    validation_dataset=validation_dataset,
    validation_fn=validate,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE,
)
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

#! [inference]

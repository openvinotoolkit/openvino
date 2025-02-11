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
validation_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [validation]
import numpy as np
import torch
from sklearn.metrics import accuracy_score

import openvino


def validate(model: openvino.CompiledModel, 
             validation_loader: torch.utils.data.DataLoader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for images, target in validation_loader:
        pred = model(images)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)
#! [validation]

#! [quantization]
model = ... # openvino.Model object

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

# compile the model to transform quantized operations to int8
model_int8 = ov.compile_model(quantized_model)

input_fp32 = ... # FP32 model input
res = model_int8(input_fp32)
#! [inference]

#! [save]
# save the model with compress_to_fp16=False to avoid an accuracy drop from compression
# of unquantized weights to FP16. This is necessary because
# nncf.quantize_with_accuracy_control(...) keeps the most impactful operations within
# the model in the original precision to achieve the specified model accuracy
ov.save_model(quantized_model, "quantized_model.xml", compress_to_fp16=False)
#! [save]

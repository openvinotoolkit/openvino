# Copyright (C) 2018-2022 Intel Corporation
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
model = ... # onnx.ModelProto object

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]
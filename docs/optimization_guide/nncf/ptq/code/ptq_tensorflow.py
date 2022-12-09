# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [dataset]
import nncf
import tensorflow_datasets as tfds

calibration_loader = tfds.load(...)

def transform_fn(data_item):
    images, _ = data_item
    return images

calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
#! [dataset]

#! [quantization]
model = ... # tensorflow.Module object

quantized_model = nncf.quantize(model, calibration_dataset)
#! [quantization]
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx

from openvino.runtime import Core, Tensor, Model


def import_onnx_model(model: onnx.ModelProto) -> Model:
    onnx.checker.check_model(model)
    model_byte_string = model.SerializeToString()
    core = Core()
    model = core.read_model(bytes(model_byte_string), Tensor(type=np.uint8, shape=[]))

    return model

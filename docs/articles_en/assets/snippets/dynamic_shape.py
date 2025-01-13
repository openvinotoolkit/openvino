# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from snippets import get_model

model = get_model()

#! [static_shape]
import openvino as ov

core = ov.Core()
model.reshape([10, 20, 30, 40])
#! [static_shape]

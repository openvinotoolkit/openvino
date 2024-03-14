# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from utils import get_path_to_model

device = "CPU"
model_path = get_path_to_model()
properties = {}

#! [export_compiled_model]

import openvino as ov

core = ov.Core()

compiled_model = core.compile_model(model_path, device, properties)
output_stream = compiled_model.export_model()
#! [export_compiled_model]

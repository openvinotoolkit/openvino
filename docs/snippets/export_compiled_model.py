# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

device = "CPU"

#! [export_compiled_model]

import openvino as ov

ov.Core().compile_model(device, modelPath, properties).export_model(compiled_blob)

#! [export_compiled_model]

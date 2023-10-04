# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from snippets import get_model

model = get_model()

#! [compile_model_default]
import openvino as ov

core = ov.Core()
compiled_model = core.compile_model(model, "CPU")
#! [compile_model_default]

#! [compile_model_multi]
core = ov.Core()
compiled_model = core.compile_model(model, "MULTI:CPU,GPU.0")
#! [compile_model_multi]

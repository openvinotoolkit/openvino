# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


#! [compile_model_default]
from openvino.runtime import Core

core = Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "CPU")
#! [compile_model_default]

#! [compile_model_multi]
core = Core()
model = core.read_model("model.xml")
compiled_model = core.compile_model(model, "MULTI:CPU,GPU.0")
#! [compile_model_multi]

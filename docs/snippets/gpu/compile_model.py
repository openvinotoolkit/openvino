# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import openvino as ov

#! [compile_model_default_gpu]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "GPU")
#! [compile_model_default_gpu]

#! [compile_model_gpu_with_id]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "GPU.1")
#! [compile_model_gpu_with_id]

#! [compile_model_gpu_with_id_and_tile]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "GPU.1.0")
#! [compile_model_gpu_with_id_and_tile]

#! [compile_model_multi]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "MULTI:GPU.1,GPU.0")
#! [compile_model_multi]

#! [compile_model_batch_plugin]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "BATCH:GPU")
#! [compile_model_batch_plugin]

#! [compile_model_auto_batch]
core = ov.Core()
model = core.read_model("model.xml")
compiled_model = ov.compile_model(model, "GPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
#! [compile_model_auto_batch]

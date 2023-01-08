# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
from openvino.runtime import Core, set_batch
from openvino.preprocess import PrePostProcessor
#! [import]

model_path = "model.xml"
batch_size = 8

#! [ov_gna_read_model]
core = Core()
model = core.read_model(model=model_path)
#! [ov_gna_read_model]

#! [ov_gna_set_nc_layout]
ppp = PrePostProcessor(model)
for i in range(len(model.inputs)):
    input_name = model.input(i).get_any_name()
    ppp.input(i).model().set_layout("N?")
model = ppp.build()
#! [ov_gna_set_nc_layout]

#! [ov_gna_set_batch_size]
set_batch(model, batch_size)
#! [ov_gna_set_batch_size]
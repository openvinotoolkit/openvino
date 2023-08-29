# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
import openvino as ov
#! [import]

batch_size = 8

# TODO: model is only available as function from snippets
#! [ov_gna_read_model]
core = ov.Core()

from snippets import get_model

model = get_model(input_shape=[1, 32])
#! [ov_gna_read_model]

#! [ov_gna_set_nc_layout]
ppp = ov.preprocess.PrePostProcessor(model)
for i in range(len(model.inputs)):
    input_name = model.input(i).get_any_name()
    ppp.input(i).model().set_layout("N?")
model = ppp.build()
#! [ov_gna_set_nc_layout]

#! [ov_gna_set_batch_size]
ov.set_batch(model, batch_size)
#! [ov_gna_set_batch_size]

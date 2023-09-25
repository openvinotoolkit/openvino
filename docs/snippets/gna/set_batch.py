# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
import openvino as ov
#! [import]

from snippets import get_path_to_model

def main():
    batch_size = 8
    model_path = get_path_to_model([1, 32])

    core = ov.Core()
    if "GNA" not in core.available_devices:
        return 0

    #! [ov_gna_read_model]
    core = ov.Core()
    model = core.read_model(model=model_path)
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

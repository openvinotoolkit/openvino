# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
from openvino.runtime import Core
#! [import]

model_path = "model.xml"

#! [ov_gna_exec_mode_hw_with_sw_fback]
core = Core()
model = core.read_model(model=model_path)
compiled_model = core.compile_model(model, device_name="GNA",
    config={ 'GNA_DEVICE_MODE' : 'GNA_HW_WITH_SW_FBACK'})
#! [ov_gna_exec_mode_hw_with_sw_fback]
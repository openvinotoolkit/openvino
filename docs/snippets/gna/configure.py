# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
import openvino as ov
#! [import]

from snippets import get_model

def main():
    model = get_model()

    core = ov.Core()
    if "GNA" not in core.available_devices:
        return 0

    # TODO: no GNA properties to replace strings
    #! [ov_gna_exec_mode_hw_with_sw_fback]
    core = ov.Core()
    compiled_model = core.compile_model(
        model, device_name="GNA", config={"GNA_DEVICE_MODE": "GNA_HW_WITH_SW_FBACK"}
    )
    #! [ov_gna_exec_mode_hw_with_sw_fback]

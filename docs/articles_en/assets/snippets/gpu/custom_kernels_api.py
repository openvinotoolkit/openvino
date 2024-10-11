# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino as ov


def main():
    core = ov.Core()

    if "GPU" not in core.available_devices:
        return 0

    # TODO: missing property to replace string
    #! [part0]
    core = ov.Core()
    core.set_property("GPU", {"CONFIG_FILE": "<path_to_the_xml_file>"})
    #! [part0]

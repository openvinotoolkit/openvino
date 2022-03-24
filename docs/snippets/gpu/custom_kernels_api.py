# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.runtime import Core

#! [part0]
core = Core()
core.set_property("GPU", {"CONFIG_FILE": "<path_to_the_xml_file>"})
#! [part0]

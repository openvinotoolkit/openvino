# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov

#! [add_extension]
# Not implemented
#! [add_extension]

#! [add_frontend_extension]
# Not implemented
#! [add_frontend_extension]

#! [add_extension_lib]
core = ov.Core()
# Load extensions library to ov.Core
core.add_extension("openvino_template_extension.so")
#! [add_extension_lib]


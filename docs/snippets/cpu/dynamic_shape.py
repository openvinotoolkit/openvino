# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.runtime import Core

#! [defined_upper_bound]
core = Core()
model = core.read_model("model.xml")
model.reshape([(1, 10), (1, 20), (1, 30), (1, 40)])
#! [defined_upper_bound]

#! [static_shape]
core = Core()
model = core.read_model("model.xml")
model.reshape([10, 20, 30, 40])
#! [static_shape]

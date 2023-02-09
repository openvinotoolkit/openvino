# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from openvino.runtime import Core

#! [static_shape]
core = Core()
model = core.read_model("model.xml")
model.reshape([10, 20, 30, 40])
#! [static_shape]

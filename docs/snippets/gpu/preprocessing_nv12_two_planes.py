# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [init_preproc]
from openvino.runtime import Core, Type, Layout
from openvino.preprocess import PrePostProcessor, ColorFormat

core = Core()
model = core.read_model("model.xml")

p = PrePostProcessor(model)
p.input().tensor().set_element_type(Type.u8) \
                  .set_color_format(ColorFormat.NV12_TWO_PLANES, ["y", "uv"]) \
                  .set_memory_type("GPU_SURFACE")
p.input().preprocess().convert_color(ColorFormat.BGR)
p.input().model().set_layout(Layout("NCHW"))
model_with_preproc = p.build()
#! [init_preproc]

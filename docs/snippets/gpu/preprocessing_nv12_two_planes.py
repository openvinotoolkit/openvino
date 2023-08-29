# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from snippets import get_model

model = get_model()

#! [init_preproc]
import openvino as ov
from openvino.preprocess import PrePostProcessor, ColorFormat

core = ov.Core()

p = PrePostProcessor(model)
p.input().tensor().set_element_type(ov.Type.u8).set_color_format(
    ColorFormat.NV12_TWO_PLANES, ["y", "uv"]
).set_memory_type("GPU_SURFACE")
p.input().preprocess().convert_color(ColorFormat.BGR)
p.input().model().set_layout(ov.Layout("NCHW"))
model_with_preproc = p.build()
#! [init_preproc]

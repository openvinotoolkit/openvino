# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from utils import get_model, get_ngraph_model

#! [ov_imports]
from openvino import Core, Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
#! [ov_imports]

#! [imports]
import openvino.inference_engine as ie
#! [imports]

#include "inference_engine.hpp"

tensor_name="input"
core = Core()
model = get_model([1,32,32,3])

#! [ov_mean_scale]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
# we only need to know where is C dimension
input.model().set_layout(Layout('...C'))
# specify scale and mean values, order of operations is important
input.preprocess().mean([116.78]).scale([57.21, 57.45, 57.73])
# insert preprocessing operations to the 'model'
model = ppp.build()
#! [ov_mean_scale]

model = get_model()

#! [ov_conversions]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
input.tensor().set_layout(Layout('NCHW')).set_element_type(Type.u8)
input.model().set_layout(Layout('NCHW'))
# layout and precision conversion is inserted automatically,
# because tensor format != model input format
model = ppp.build()
#! [ov_conversions]

#! [ov_color_space]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
input.tensor().set_color_format(ColorFormat.NV12_TWO_PLANES)
# add NV12 to BGR conversion
input.preprocess().convert_color(ColorFormat.BGR)
# and insert operations to the model
model = ppp.build()
#! [ov_color_space]

model = get_model([1, 3, 448, 448])

#! [ov_image_scale]
ppp = PrePostProcessor(model)
input = ppp.input("input")
# need to specify H and W dimensions in model, others are not important
input.model().set_layout(Layout('??HW'))
# scale to model shape
input.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, 448, 448)
# and insert operations to the model
model = ppp.build()
#! [ov_image_scale]

import openvino.inference_engine as ie
import ngraph as ng

operation_name = "data"

core = ie.IECore()
network = get_ngraph_model()


#! [mean_scale]
preprocess_info = network.input_info[operation_name].preprocess_info
preprocess_info.init(3)
preprocess_info[0].mean_value = 116.78
preprocess_info[1].mean_value = 116.78
preprocess_info[2].mean_value = 116.78
preprocess_info[0].std_scale = 57.21
preprocess_info[1].std_scale = 57.45
preprocess_info[2].std_scale = 57.73
preprocess_info.mean_variant = ie.MeanVariant.MEAN_VALUE
#! [mean_scale]

#! [conversions]
input_info = network.input_info[operation_name]
input_info.precision = "U8"
input_info.layout = "NHWC"
# model input layout is always NCHW in Inference Engine
# for shapes with 4 dimensions
#! [conversions]

#! [image_scale]
preprocess_info = network.input_info[operation_name].preprocess_info
# Inference Engine supposes input for resize is always in NCHW layout
# while for OpenVINO Runtime API 2.0 `H` and `W` dimensions must be specified
# Also, current code snippet supposed resize from dynamic shapes
preprocess_info.resize_algorithm = ie.ResizeAlgorithm.RESIZE_BILINEAR

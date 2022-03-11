# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#! [ov_imports]
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm
#! [ov_imports]

#! [imports]
import openvino.inference_engine as ie
#! [imports]

#include "inference_engine.hpp"

model_path = ''
tensor_name = ''

core = Core()
model = core.read_model(model=model_path)

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

#! [ov_image_scale]
ppp = PrePostProcessor(model)
input = ppp.input(tensor_name)
# need to specify H and W dimensions in model, others are not important
input.model().set_layout(Layout('??HW'))
# scale to model shape
input.preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR, 448, 448)
# and insert operations to the model
model = ppp.build()
#! [ov_image_scale]



model_path = ''
operation_name = ''

core = Core()
network = core.ReadNetwork(model_path)


#! [mean_scale]
preProcess = network.getInputsInfo()[operation_name].getPreProcess()
preProcess.init(3)
preProcess[0].meanValue = 116.78
preProcess[1].meanValue = 116.78
preProcess[2].meanValue = 116.78
preProcess[0].stdScale = 57.21
preProcess[1].stdScale = 57.45
preProcess[2].stdScale = 57.73
preProcess.setVariant(ie.MEAN_VALUE)
#! [mean_scale]

#! [conversions]
inputInfo = network.getInputsInfo()[operation_name]
inputInfo.setPrecision(ie.Precision.U8)
inputInfo.setLayout(ie.Layout.NHWC)
# model input layout is always NCHW in Inference Engine
# for shapes with 4 dimensions
#! [conversions]

#! [color_space]

preProcess = network.getInputsInfo()[operation_name].getPreProcess()
# Inference Engine supposes NV12 as two inputs which need to be passed
# as InferenceEngine::NV12Blob composed of two Y and UV planes
preProcess.setColorFormat(ie.NV12)
#! [color_space]

#! [image_scale]
preProcess = network.getInputsInfo()[operation_name].getPreProcess()
# Inference Engine supposes input for resize is always in NCHW layout
# while for OpenVINO Runtime API 2.0 `H` and `W` dimensions must be specified
# Also, current code snippet supposed resize from dynamic shapes
preProcess.setResizeAlgorithm(ie.ResizeAlgorithm.RESIZE_BILINEAR)
#! [image_scale]

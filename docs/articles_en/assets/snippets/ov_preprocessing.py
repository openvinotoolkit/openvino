# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.properties as props
from openvino.preprocess import ResizeAlgorithm, ColorFormat
from openvino import Layout, Type, serialize
import openvino as ov
import openvino.runtime.opset12 as ops
from utils import get_model, get_temp_dir, get_path_to_model, get_advanced_model


input_name = "input"
model = get_advanced_model()

# ! [ov:preprocess:create]
from openvino.preprocess import PrePostProcessor

ppp = PrePostProcessor(model)
# ! [ov:preprocess:create]

# ! [ov:preprocess:tensor]
from openvino.preprocess import ColorFormat
from openvino import Layout, Type
ppp.input(input_name).tensor() \
        .set_element_type(Type.u8) \
        .set_shape([1, 480, 640, 3]) \
        .set_layout(Layout('NHWC')) \
        .set_color_format(ColorFormat.BGR)
# ! [ov:preprocess:tensor]
# ! [ov:preprocess:model]
# `model's input` already `knows` it's shape and data type, no need to specify them here
ppp.input(input_name).model().set_layout(Layout('NCHW'))
# ! [ov:preprocess:model]
# ! [ov:preprocess:steps]
from openvino.preprocess import ResizeAlgorithm
ppp.input(input_name).preprocess() \
    .convert_element_type(Type.f32) \
    .convert_color(ColorFormat.RGB) \
    .resize(ResizeAlgorithm.RESIZE_LINEAR) \
    .mean([100.5, 101, 101.5]) \
    .scale([50., 51., 52.])
# .convert_layout(Layout('NCHW')) # Not needed, such conversion will be added implicitly
# ! [ov:preprocess:steps]
# ! [ov:preprocess:build]
print(f'Dump preprocessor: {ppp}')
model = ppp.build()
# ! [ov:preprocess:build]

# ! [ov:preprocess:input_index]
ppp.input(1) # Gets 2nd input in a model
ppp.output(2) # Gets output with index=2 (3rd one) in a model
# ! [ov:preprocess:input_index]


# ! [ov:preprocess:input_name]
ppp.input(input_name)
ppp.output('result')
# ! [ov:preprocess:input_name]

model = get_model()
ppp = PrePostProcessor(model)

# ! [ov:preprocess:input_1]
# no index/name is needed if model has one input
ppp.input().preprocess().scale(50.)

# same for output
ppp.output() \
    .postprocess().convert_element_type(Type.u8)
# ! [ov:preprocess:input_1]
# ! [ov:preprocess:mean_scale]
ppp.input(input_name).preprocess().mean(128).scale(127)
# ! [ov:preprocess:mean_scale]
# ! [ov:preprocess:mean_scale_array]
# Suppose model's shape is {1, 3, 224, 224}
# N=1, C=3, H=224, W=224
ppp.input(input_name).model().set_layout(Layout('NCHW'))
# Mean/Scale has 3 values which matches with C=3
ppp.input(input_name).preprocess() \
    .mean([103.94, 116.78, 123.68]).scale([57.21, 57.45, 57.73])
# ! [ov:preprocess:mean_scale_array]
# ! [ov:preprocess:convert_element_type]
# First define data type for your tensor
ppp.input(input_name).tensor().set_element_type(Type.u8)

# Then define preprocessing step
ppp.input(input_name).preprocess().convert_element_type(Type.f32)

# If conversion is needed to `model's` element type, 'f32' can be omitted
ppp.input(input_name).preprocess().convert_element_type()
# ! [ov:preprocess:convert_element_type]
# ! [ov:preprocess:convert_layout]
# First define layout for your tensor
ppp.input(input_name).tensor().set_layout(Layout('NHWC'))

# Then define layout of model
ppp.input(input_name).model().set_layout(Layout('NCHW'))

print(ppp)  # Will print 'implicit layout conversion step'
# ! [ov:preprocess:convert_layout]
# ! [ov:preprocess:convert_layout_2]
ppp.input(input_name).tensor().set_shape([1, 480, 640, 3])

# Model expects shape {1, 3, 480, 640}
ppp.input(input_name).preprocess()\
    .convert_layout([0, 3, 1, 2])
# 0 -> 0; 3 -> 1; 1 -> 2; 2 -> 3
# ! [ov:preprocess:convert_layout_2]

# ! [ov:preprocess:resize_1]
ppp.input(input_name).tensor().set_shape([1, 3, 960, 1280])
ppp.input(input_name).model().set_layout(Layout('??HW'))
ppp.input(input_name).preprocess()\
    .resize(ResizeAlgorithm.RESIZE_LINEAR, 480, 640)
# ! [ov:preprocess:resize_1]
# ! [ov:preprocess:resize_2]
ppp.input(input_name).tensor().set_shape([1, 3, 960, 1280])
# Model accepts {1, 3, 480, 640} shape, thus last dimensions are 'H' and 'W'
ppp.input(input_name).model().set_layout(Layout('??HW'))
# Resize to model's dimension
ppp.input(input_name).preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
# ! [ov:preprocess:resize_2]
# ! [ov:preprocess:convert_color_1]
ppp.input(input_name).tensor().set_color_format(ColorFormat.BGR)

ppp.input(input_name).preprocess().convert_color(ColorFormat.RGB)
# ! [ov:preprocess:convert_color_1]
# ! [ov:preprocess:convert_color_2]
# This will split original `input` to 2 separate inputs: `input/y' and 'input/uv'
ppp.input(input_name).tensor()\
    .set_color_format(ColorFormat.NV12_TWO_PLANES)

ppp.input(input_name).preprocess()\
    .convert_color(ColorFormat.RGB)
print(ppp)  # Dump preprocessing steps to see what will happen
# ! [ov:preprocess:convert_color_2]

# ! [ov:preprocess:custom]
# It is possible to insert some custom operations
import openvino.runtime.opset12 as ops
from openvino.runtime import Output
from openvino.runtime.utils.decorators import custom_preprocess_function

@custom_preprocess_function
def custom_abs(output: Output):
    # Custom nodes can be inserted as Preprocessing steps
    return ops.abs(output)

ppp.input("input").preprocess() \
    .custom(custom_abs)
# ! [ov:preprocess:custom]

# ! [ov:preprocess:postprocess]
# Model's output has 'NCHW' layout
ppp.output('result').model().set_layout(Layout('NCHW'))

# Set target user's tensor to U8 type + 'NHWC' layout
# Precision & layout conversions will be done implicitly
ppp.output('result').tensor()\
    .set_layout(Layout("NHWC"))\
    .set_element_type(Type.u8)

# Also it is possible to insert some custom operations
import openvino.runtime.opset12 as ops
from openvino.runtime import Output
from openvino.runtime.utils.decorators import custom_preprocess_function

@custom_preprocess_function
def custom_abs(output: Output):
    # Custom nodes can be inserted as Post-processing steps
    return ops.abs(output)

ppp.output("result").postprocess()\
    .custom(custom_abs)
# ! [ov:preprocess:postprocess]

# ! [ov:preprocess:save_headers]
from openvino.preprocess import PrePostProcessor, ColorFormat, ResizeAlgorithm
from openvino import Core, Layout, Type, set_batch
from openvino import serialize
# ! [ov:preprocess:save_headers]

model_path = get_path_to_model()
serialized_model_path = get_path_to_model()

# ! [ov:preprocess:save_model]
# ========  Step 0: read original model =========
core = Core()
model = core.read_model(model=model_path)

# ======== Step 1: Preprocessing ================
ppp = PrePostProcessor(model)
# Declare section of desired application's input format
ppp.input().tensor() \
    .set_element_type(Type.u8) \
    .set_spatial_dynamic_shape() \
    .set_layout(Layout('NHWC')) \
    .set_color_format(ColorFormat.BGR)

# Specify actual model layout
ppp.input().model().set_layout(Layout('NCHW'))

# Explicit preprocessing steps. Layout conversion will be done automatically as last step
ppp.input().preprocess() \
    .convert_element_type() \
    .convert_color(ColorFormat.RGB) \
    .resize(ResizeAlgorithm.RESIZE_LINEAR) \
    .mean([123.675, 116.28, 103.53]) \
    .scale([58.624, 57.12, 57.375])

# Dump preprocessor
print(f'Dump preprocessor: {ppp}')
model = ppp.build()

# ======== Step 2: Change batch size ================
# In this example we also want to change batch size to increase throughput
set_batch(model, 2)

# ======== Step 3: Save the model ================
serialize(model, serialized_model_path)
# ! [ov:preprocess:save_model]

path_to_cache_dir = get_temp_dir()

# ! [ov:preprocess:save_load]
core = Core()
core.set_property({props.cache_dir(): path_to_cache_dir})

# In case that no preprocessing is needed anymore, we can load model on target device directly
# With cached model available, it will also save some time on reading original model
compiled_model = core.compile_model(serialized_model_path, 'CPU')
# ! [ov:preprocess:save_load]

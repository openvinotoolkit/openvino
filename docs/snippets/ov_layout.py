# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import openvino.runtime as ov

# ! [ov:layout:simple]
from openvino.runtime import Layout
layout = Layout('NCHW')
# ! [ov:layout:simple]
# ! [ov:layout:complex]
# Each dimension has name separated by comma
# Layout is wrapped with square brackets
layout = Layout('[time,temperature,humidity]')
# ! [ov:layout:complex]
# ! [ov:layout:partially_defined]
# First dimension is batch, 4th is 'channels'.
# Others are not important for us
layout = Layout('N??C')

# Or the same using advanced syntax
layout = Layout('[n,?,?,c]')
# ! [ov:layout:partially_defined]
# ! [ov:layout:dynamic]
# First dimension is 'batch' others are whatever
layout = Layout('N...')

# Second dimension is 'channels' others are whatever
layout = Layout('?C...')

# Last dimension is 'channels' others are whatever
layout = Layout('...C')
# ! [ov:layout:dynamic]

# ! [ov:layout:predefined]
from openvino.runtime import layout_helpers
# returns 0 for batch
layout_helpers.batch_idx(Layout('NCDHW'))

# returns 1 for channels
layout_helpers.channels_idx(Layout('NCDHW'))

# returns 2 for depth
layout_helpers.depth_idx(Layout('NCDHW'))

# returns -2 for height
layout_helpers.height_idx(Layout('...HW'))

# returns -1 for width
layout_helpers.width_idx(Layout('...HW'))
# ! [ov:layout:predefined]

# ! [ov:layout:dump]
layout = Layout('NCHW')
print(layout)    # prints [N,C,H,W]
# ! [ov:layout:dump]


def create_simple_model():
    # This example shows how to create ov::Function
    #
    # Parameter--->Multiply--->Add--->Result
    #    Constant---'          /
    #              Constant---'
    data = ov.opset8.parameter([3, 1, 2], ov.Type.f32)
    mul_constant = ov.opset8.constant([1.5], ov.Type.f32)
    mul = ov.opset8.multiply(data, mul_constant)
    add_constant = ov.opset8.constant([0.5], ov.Type.f32)
    add = ov.opset8.add(mul, add_constant)
    res = ov.opset8.result(add)
    return ov.Model([res], [data], "model")

model = create_simple_model()

# ! [ov:layout:get_from_model]
# Get layout for model input
layout = layout_helpers.get_layout(model.input("input_tensor_name"))
# Get layout for model with single output
layout = layout_helpers.get_layout(model.output())
# ! [ov:layout:get_from_model]

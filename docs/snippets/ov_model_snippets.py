# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino as ov
#! [import]
import openvino.runtime.opset12 as ops
#! [import]
import openvino.runtime.passes as passes


# ! [ov:create_simple_model]
def create_simple_model():
    # This example shows how to create ov::Function
    #
    # Parameter--->Multiply--->Add--->Result
    #    Constant---'          /
    #              Constant---'
    data = ops.parameter([3, 1, 2], ov.Type.f32)
    mul_constant = ops.constant([1.5], ov.Type.f32)
    mul = ops.multiply(data, mul_constant)
    add_constant = ops.constant([0.5], ov.Type.f32)
    add = ops.add(mul, add_constant)
    res = ops.result(add)
    return ov.Model([res], [data], "model")
# ! [ov:create_simple_model]

# ! [ov:create_advanced_model]
def create_advanced_model():
    # Advanced example with multi output operation
    #
    # Parameter->Split---0-->Result
    #               | `--1-->Relu-->Result
    #               `----2-->Result
    data = ops.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32)
    # Create Constant for axis value
    axis_const = ops.constant(1, dtype=ov.Type.i64)

    # Create opset12::Split operation that splits input to three slices across 1st dimension
    split = ops.split(data, axis_const, 3)

    # Create opset12::Relu operation that takes 1st Split output as input
    relu = ops.relu(split.output(1))

    # Results operations will be created automatically based on provided OutputVector
    return ov.Model([split.output(0), relu.output(0), split.output(2)], [data], "model")
# ! [ov:create_advanced_model]

def ov_api_examples():
    # Doesn't work
    # node = ov.opset8.parameter(ov.PartialShape([ov.Dimension.dynamic(), 3, 64, 64]), np.float32)
    node = ops.parameter(ov.PartialShape([ov.Dimension.dynamic(), ov.Dimension(3), ov.Dimension(64), ov.Dimension(64)]), np.float32)

    # it doesn't work:
    # static_shape = ov.Shape()
    # ! [ov:partial_shape]
    partial_shape = node.output(0).get_partial_shape() # get zero output partial shape
    if not partial_shape.is_dynamic: # or partial_shape.is_static
        static_shape = partial_shape.get_shape()
    # ! [ov:partial_shape]

# ! [ov:serialize]
def serialize_example(m : ov.Model):
    ov.serialize(m, xml_path='model.xml', bin_path='model.bin')
# ! [ov:serialize]

# ! [ov:visualize]
def visualize_example(m : ov.Model):
    # Need import:
    # * import openvino.runtime.passes as passes
    pass_manager = passes.Manager()
    pass_manager.register_pass(passes.VisualizeTree(file_name='image.svg'))
    pass_manager.run_passes(m)
# ! [ov:visualize]

def model_inputs_outputs(model : ov.Model):
    #! [all_inputs_ouputs]
    inputs = model.inputs
    outputs = model.outputs
    #! [all_inputs_ouputs]


def main():
    ov_api_examples()
    create_simple_model()
    model = create_advanced_model()
    serialize_example(model)
    visualize_example(model)
    model_inputs_outputs(model)

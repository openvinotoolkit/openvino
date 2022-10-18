# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
#! [import]
import openvino.runtime as ov
#! [import]
import openvino.runtime.passes as passes

# ! [ov:create_simple_model]
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
# ! [ov:create_simple_model]

# ! [ov:create_advanced_model]
def create_advanced_model():
    # Advanced example with multi output operation
    #
    # Parameter->Split---0-->Result
    #               | `--1-->Relu-->Result
    #               `----2-->Result
    data = ov.opset8.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32)
    # Create Constant for axis value
    axis_const = ov.opset8.constant(ov.Type.i64, ov.Shape({}), [1])

    # Create opset8::Split operation that splits input to three slices across 1st dimension
    split = ov.opset8.split(data, axis_const, 3)

    # Create opset8::Relu operation that takes 1st Split output as input
    relu = ov.opset8.relu(split.output(1))

    # Results operations will be created automatically based on provided OutputVector
    return ov.Model([split.output(0), relu, split.output[2]], [data], "model")
# ! [ov:create_advanced_model]

def ov_api_examples():
    # Doesn't work
    # node = ov.opset8.parameter(ov.PartialShape([ov.Dimension.dynamic(), 3, 64, 64]), np.float32)
    node = ov.opset8.parameter(ov.PartialShape([ov.Dimension.dynamic(), ov.Dimension(3), ov.Dimension(64), ov.Dimension(64)]), np.float32)

    # it doesn't work:
    # static_shape = ov.Shape()
    # ! [ov:partial_shape]
    partial_shape = node.output(0).get_partial_shape() # get zero output partial shape
    if not partial_shape.is_dynamic: # or partial_shape.is_static
        static_shape = partial_shape.get_shape()
    # ! [ov:partial_shape]

# ! [ov:serialize]
def serialize_example(m : ov.Model):
    # Need import:
    # * import openvino.runtime.passes as passes
    pass_manager = passes.Manager()
    pass_manager.register_pass(pass_name="Serialize", xml_path='model.xml', bin_path='model.bin')
    pass_manager.run_passes(m)
# ! [ov:serialize]

# ! [ov:visualize]
def visualize_example(m : ov.Model):
    # Need import:
    # * import openvino.runtime.passes as passes
    pass_manager = passes.Manager()
    pass_manager.register_pass(pass_name="VisualTree", file_name='image.svg')
    pass_manager.run_passes(m)
# ! [ov:visualize]

def model_inputs_outputs(model : ov.Model):
    #! [all_inputs_ouputs]
    inputs = model.inputs
    outputs = model.outputs
    #! [all_inputs_ouputs]


if __name__ == '__main__':
    ov_api_examples()
    create_simple_model()
    create_advanced_model()

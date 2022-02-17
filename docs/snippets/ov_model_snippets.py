# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#! [import]
import openvino.runtime as ov
#! [import]

# ! [ov:create_simple_model]
def create_simple_model():
    # This example shows how to create ov::Function
    #
    # Parameter--->Multiply--->Add--->Result
    #    Constant---'          /
    #              Constant---'
    data = ov.opset8.parameter(ov.Shape([3, 1, 2]), ov.Type.f32)
    mul_constant = ov.opset8.constant(ov.Type.f32, ov.Shape({1}), [1.5])
    mul = ov.opset8.multiply(data, mul_constant)
    add_constant = ov.opset8.constant(ov.Type.f32, ov.Shape({1}), [0.5])
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
    data = ov.opset8.parameter(ov.Shape[1, 3, 64, 64], ov.Type.f32)
    # Create Constant for axis value
    axis_const = ov.opset8.constant(ov.Type.i64, ov.Shape({}), [1])

    # Create opset8::Split operation that splits input to three slices across 1st dimension
    split = ov.opset8.split(data, axis_const, 3)

    # Create opset8::Relu operation that takes 1st Split output as input
    relu = ov.opset8.relu(split.output(1))

    # Results operations will be created automatically based on provided OutputVector
    return ov.Model([split.output(0), relu, split.output[2]], [data], "model")
# ! [ov:create_advanced_model]

if __name__ == '__main__':
    create_simple_model()
    create_advanced_model()

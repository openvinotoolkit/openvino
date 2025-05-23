# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
from openvino.op import Parameter, Constant
from openvino.opset13 import add, multiply

import openvino as ov
from tests.utils.helpers import create_filenames_for_ir


def make_constant(values, transposed):
    return Constant(ov.Type.f32, ov.Shape([1, len(values)] if transposed else [len(values), 1]), values)


# keep fp16 denormals, flush fp32 denormals to zero
in_range = [-65504.0, -2.0, 1.00097656, -1.0, -0.99951172, -0.00006103515625, -0.000000059604645, 0.0,
            0.000000059604645, 0.99951172, 0.00006103515625, 1.0, 1.00097656, 2.0, 65504]
out_of_range = [float("-inf"), -65505.0, -1e-10, -1e-39, 1e-39, 1e-10, 65505.0, float("inf")]
converted_out_of_range = [-65504.0, -65504.0, 0, 0, 0, 0, 65504.0, 65504.0]

# test inputs
more_in_range = out_of_range + 10 * in_range
more_out_of_range = in_range + 10 * out_of_range

# reference after conversion more_in_range to fp16
converted_more_in_range = converted_out_of_range + 10 * in_range


def make_model(add_consts, mul_consts):
    parameter1 = Parameter(ov.Type.f32, ov.PartialShape([-1]))
    add1 = add(parameter1, make_constant(add_consts, False))
    mul1 = multiply(add1, make_constant(mul_consts, True))
    return ov.Model([mul1], [parameter1])


def get_constants(model, request, tmp_path) -> List[Constant]:
    model_fname, _ = create_filenames_for_ir(request.node.name, tmp_path)
    ov.save_model(model, model_fname)
    core = ov.Core()
    restored_model = core.read_model(model_fname)

    op_ind_map = {"Add": 0, "Multiply": 1}
    constants_list = [[]] * len(op_ind_map)

    for op in restored_model.get_ordered_ops():
        op_type = op.get_type_info().name
        if op_type not in op_ind_map.keys():
            continue

        in_node = op.input_value(1).get_node()
        if in_node.get_type_info().name == "Convert":
            const_node = in_node.input_value(0).get_node()
            if const_node.get_type_info().name != "Constant":
                const_node = None
        elif in_node.get_type_info().name == "Constant":
            const_node = in_node

        constants_list[op_ind_map[op_type]] = const_node

    for node in constants_list:
        assert not isinstance(node, list)

    # sanity check that model is compilable
    ov.compile_model(restored_model)
    return constants_list


def test_compression_1(request, tmp_path):
    model = make_model(more_in_range, more_out_of_range)
    const_fp16, const_fp32 = get_constants(model, request, tmp_path)
    assert const_fp32 is not None, "There is no Constant op on FP32 branch"
    assert const_fp16 is not None, "There is no compressed Constant + Convert op on FP16 branch"

    assert const_fp32.get_output_element_type(0) == ov.Type.f32
    assert np.all(np.array(more_out_of_range, dtype=np.float32) == const_fp32.get_vector())

    assert const_fp16.get_output_element_type(0) == ov.Type.f16

    msg = f"Difference: {np.array(converted_more_in_range, dtype=np.float32) - const_fp16.get_vector()}"
    assert np.all(np.array(converted_more_in_range, dtype=np.float32) == const_fp16.get_vector()), msg


def test_compression_2(request, tmp_path):
    model = make_model(more_in_range, more_in_range)
    const_fp16_1, const_fp16_2 = get_constants(model, request, tmp_path)

    assert const_fp16_1 is not None, "There is no Constant op on FP16 branch"
    assert const_fp16_2 is not None, "There is no Constant op on FP16 branch"

    assert const_fp16_1.get_output_element_type(0) == ov.Type.f16, "Const element type is not f16"
    assert const_fp16_2.get_output_element_type(0) == ov.Type.f16, "Const element type is not f16"
    f16_min, f16_max = np.finfo(np.float16).min, np.finfo(np.float16).max
    in_range_clipped = np.clip(more_in_range, f16_min, f16_max).astype(np.float16)

    assert np.all(in_range_clipped == const_fp16_1.get_vector())
    assert np.all(in_range_clipped == const_fp16_2.get_vector())


def test_no_compression(request, tmp_path):
    model = make_model(more_out_of_range, more_out_of_range)
    const_fp32_1, const_fp32_2 = get_constants(model, request, tmp_path)

    assert const_fp32_1 is not None, "There is no Constant op on FP32 branch"
    assert const_fp32_2 is not None, "There is no Constant op on FP32 branch"

    assert const_fp32_1.get_output_element_type(0) == ov.Type.f32, "Const element type is not f32"

    assert const_fp32_2.get_output_element_type(0) == ov.Type.f32, "Const element type is not f32"

    assert np.all(np.array(more_out_of_range, dtype=np.float32) == const_fp32_1.get_vector())
    assert np.all(np.array(more_out_of_range, dtype=np.float32) == const_fp32_2.get_vector())

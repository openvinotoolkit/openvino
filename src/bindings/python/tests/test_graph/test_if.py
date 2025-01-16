# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import openvino.opset8 as ov
from openvino import Model

from openvino.op.util import InvariantInputDescription, BodyOutputDescription

from tests.utils.helpers import compare_models


def create_simple_if_with_two_outputs(condition_val):
    condition = ov.constant(condition_val, dtype=bool)

    # then_body
    x_t = ov.parameter([], np.float32, "X")
    y_t = ov.parameter([], np.float32, "Y")
    z_t = ov.parameter([], np.float32, "Z")

    add_t = ov.add(x_t, y_t)
    mul_t = ov.multiply(y_t, z_t)
    then_body_res_1 = ov.result(add_t)
    then_body_res_2 = ov.result(mul_t)
    then_body = Model([then_body_res_1, then_body_res_2], [x_t, y_t, z_t], "then_body_function")

    # else_body
    x_e = ov.parameter([], np.float32, "X")
    z_e = ov.parameter([], np.float32, "Z")
    w_e = ov.parameter([], np.float32, "W")

    add_e = ov.add(x_e, w_e)
    pow_e = ov.power(w_e, z_e)
    else_body_res_1 = ov.result(add_e)
    else_body_res_2 = ov.result(pow_e)
    else_body = Model([else_body_res_1, else_body_res_2], [x_e, z_e, w_e], "else_body_function")

    const_x = ov.constant(15.0, dtype=np.float32)
    const_y = ov.constant(-5.0, dtype=np.float32)
    const_z = ov.constant(4.0, dtype=np.float32)
    const_w = ov.constant(2.0, dtype=np.float32)

    if_node = ov.if_op(condition)
    if_node.set_then_body(then_body)
    if_node.set_else_body(else_body)
    if_node.set_input(const_x.output(0), x_t, x_e)
    if_node.set_input(const_y.output(0), y_t, None)
    if_node.set_input(const_z.output(0), z_t, z_e)
    if_node.set_input(const_w.output(0), None, w_e)
    if_node.set_output(then_body_res_1, else_body_res_1)
    if_node.set_output(then_body_res_2, else_body_res_2)
    return if_node


def create_diff_if_with_two_outputs(condition_val):
    condition = ov.constant(condition_val, dtype=bool)

    # then_body
    x_t = ov.parameter([2], np.float32, "X")
    y_t = ov.parameter([2], np.float32, "Y")
    mmul_t = ov.matmul(x_t, y_t, False, False)
    mul_t = ov.multiply(y_t, x_t)
    then_body_res_1 = ov.result(mmul_t)
    then_body_res_2 = ov.result(mul_t)
    then_body = Model([then_body_res_1, then_body_res_2], [x_t, y_t], "then_body_function")

    # else_body
    x_e = ov.parameter([2], np.float32, "X")
    z_e = ov.parameter([], np.float32, "Z")
    mul_e = ov.multiply(x_e, z_e)
    else_body_res_1 = ov.result(z_e)
    else_body_res_2 = ov.result(mul_e)
    else_body = Model([else_body_res_1, else_body_res_2], [x_e, z_e], "else_body_function")

    const_x = ov.constant([3, 4], dtype=np.float32)
    const_y = ov.constant([2, 1], dtype=np.float32)
    const_z = ov.constant(4.0, dtype=np.float32)

    if_node = ov.if_op(condition)
    if_node.set_then_body(then_body)
    if_node.set_else_body(else_body)
    if_node.set_input(const_x.output(0), x_t, x_e)
    if_node.set_input(const_y.output(0), y_t, None)
    if_node.set_input(const_z.output(0), None, z_e)
    if_node.set_output(then_body_res_1, else_body_res_1)
    if_node.set_output(then_body_res_2, else_body_res_2)

    return if_node


def simple_if(condition_val):
    condition = ov.constant(condition_val, dtype=bool)
    # then_body
    x_t = ov.parameter([2], np.float32, "X")
    y_t = ov.parameter([2], np.float32, "Y")

    then_mul = ov.multiply(x_t, y_t)
    then_body_res_1 = ov.result(then_mul)
    then_body = Model([then_body_res_1], [x_t, y_t], "then_body_function")

    # else_body
    x_e = ov.parameter([2], np.float32, "X")
    y_e = ov.parameter([2], np.float32, "Y")
    add_e = ov.add(x_e, y_e)
    else_body_res_1 = ov.result(add_e)
    else_body = Model([else_body_res_1], [x_e, y_e], "else_body_function")

    const_x = ov.constant([3, 4], dtype=np.float32)
    const_y = ov.constant([2, 1], dtype=np.float32)

    if_node = ov.if_op(condition)
    if_node.set_then_body(then_body)
    if_node.set_else_body(else_body)
    if_node.set_input(const_x.output(0), x_t, x_e)
    if_node.set_input(const_y.output(0), y_t, y_e)
    if_res = if_node.set_output(then_body_res_1, else_body_res_1)
    relu = ov.relu(if_res)

    return relu


def simple_if_without_parameters(condition_val):
    condition = ov.constant(condition_val, dtype=bool)

    # then_body
    then_constant = ov.constant(0.7, dtype=float)
    then_body_res_1 = ov.result(then_constant)
    then_body = Model([then_body_res_1], [])

    # else_body
    else_const = ov.constant(9.0, dtype=float)
    else_body_res_1 = ov.result(else_const)
    else_body = Model([else_body_res_1], [])

    if_node = ov.if_op(condition.output(0))
    if_node.set_then_body(then_body)
    if_node.set_else_body(else_body)
    if_res = if_node.set_output(then_body_res_1, else_body_res_1)
    relu = ov.relu(if_res)
    return relu


def check_results(results, expected_results):
    assert len(results) == len(expected_results)
    for id_result, res in enumerate(results):
        assert np.allclose(res, expected_results[id_result])


def check_if(if_model, cond_val, exp_results):
    last_node = if_model(cond_val)
    assert last_node.get_type_name() == exp_results[0]
    assert last_node.get_output_size() == exp_results[1]
    assert list(last_node.get_output_shape(0)) == exp_results[2]


def test_if_with_two_outputs():
    check_if(create_simple_if_with_two_outputs, True,
             ["If", 2, []])
    check_if(create_simple_if_with_two_outputs, False,
             ["If", 2, []])


def test_diff_if_with_two_outputs():
    check_if(create_diff_if_with_two_outputs, True,
             ["If", 2, []])
    check_if(create_diff_if_with_two_outputs, False,
             ["If", 2, []])


def test_simple_if():
    check_if(simple_if, True, ["Relu", 1, [2]])
    check_if(simple_if, False, ["Relu", 1, [2]])


def test_simple_if_without_body_parameters():
    check_if(simple_if_without_parameters, True, ["Relu", 1, []])
    check_if(simple_if_without_parameters, False, ["Relu", 1, []])


def check_if_getters(if_model, cond_val):
    if_op = if_model(cond_val)
    assert isinstance(if_op.get_then_body(), Model)
    assert if_op.get_function(0)._get_raw_address() == if_op.get_then_body()._get_raw_address()
    assert compare_models(if_op.get_function(0), if_op.get_then_body())

    assert isinstance(if_op.get_else_body(), Model)
    assert if_op.get_function(1)._get_raw_address() == if_op.get_else_body()._get_raw_address()
    assert compare_models(if_op.get_function(1), if_op.get_else_body())


@pytest.mark.parametrize(("cond_val"), [
    True,
    False,
])
def test_if_getters(cond_val):
    check_if_getters(create_simple_if_with_two_outputs, cond_val)


def test_simple_if_basic():
    condition = ov.constant(True, dtype=bool)
    # then_body
    x_t = ov.parameter([2], np.float32, "X")
    y_t = ov.parameter([2], np.float32, "Y")

    then_mul = ov.multiply(x_t, y_t)
    then_body_res_1 = ov.result(then_mul)
    then_body = Model([then_body_res_1], [x_t, y_t], "then_body_function")
    then_body_inputs = [InvariantInputDescription(1, 0), InvariantInputDescription(2, 1)]

    else_body_outputs = [BodyOutputDescription(0, 0)]

    if_node = ov.if_op(condition.output(0))
    if_node.set_function(0, then_body)
    subgraph_func = if_node.get_function(0)

    assert isinstance(subgraph_func, type(then_body))
    assert compare_models(subgraph_func, then_body)
    assert subgraph_func._get_raw_address() == then_body._get_raw_address()

    if_node.set_input_descriptions(0, then_body_inputs)
    if_node.set_output_descriptions(1, else_body_outputs)

    input_desc = if_node.get_input_descriptions(0)
    output_desc = if_node.get_output_descriptions(1)

    assert len(input_desc) == len(then_body_inputs)
    assert len(output_desc) == len(else_body_outputs)

    for i in range(len(input_desc)):
        assert input_desc[i].get_type_info() == then_body_inputs[i].get_type_info()
        assert input_desc[i].input_index == then_body_inputs[i].input_index
        assert input_desc[i].body_parameter_index == then_body_inputs[i].body_parameter_index

    for i in range(len(output_desc)):
        assert output_desc[i].get_type_info() == else_body_outputs[i].get_type_info()
        assert output_desc[i].output_index == else_body_outputs[i].output_index
        assert output_desc[i].body_value_index == else_body_outputs[i].body_value_index

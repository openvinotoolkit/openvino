# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ngraph as ng
import numpy as np
import pytest
from ngraph.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
)
from tests_compatibility import xfail_issue_00000
from tests_compatibility.runtime import get_runtime


def create_simple_if_with_two_outputs(condition_val):
    condition = ng.constant(condition_val, dtype=np.bool)

    # then_body
    X_t = ng.parameter([], np.float32, "X")
    Y_t = ng.parameter([], np.float32, "Y")
    Z_t = ng.parameter([], np.float32, "Z")

    add_t = ng.add(X_t, Y_t)
    mul_t = ng.multiply(Y_t, Z_t)
    then_body_res_1 = ng.result(add_t)
    then_body_res_2 = ng.result(mul_t)
    then_body = GraphBody([X_t, Y_t, Z_t], [then_body_res_1, then_body_res_2])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1),
                        TensorIteratorInvariantInputDesc(3, 2)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # else_body
    X_e = ng.parameter([], np.float32, "X")
    Z_e = ng.parameter([], np.float32, "Z")
    W_e = ng.parameter([], np.float32, "W")

    add_e = ng.add(X_e, W_e)
    pow_e = ng.power(W_e, Z_e)
    else_body_res_1 = ng.result(add_e)
    else_body_res_2 = ng.result(pow_e)
    else_body = GraphBody([X_e, Z_e, W_e], [else_body_res_1, else_body_res_2])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(3, 1),
                        TensorIteratorInvariantInputDesc(4, 2)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    X = ng.constant(15.0, dtype=np.float32)
    Y = ng.constant(-5.0, dtype=np.float32)
    Z = ng.constant(4.0, dtype=np.float32)
    W = ng.constant(2.0, dtype=np.float32)
    if_node = ng.if_op(condition, [X, Y, Z, W], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def create_diff_if_with_two_outputs(condition_val):
    condition = ng.constant(condition_val, dtype=np.int8)

    # then_body
    X_t = ng.parameter([2], np.float32, "X")
    Y_t = ng.parameter([2], np.float32, "Y")
    mmul_t = ng.matmul(X_t, Y_t, False, False)
    mul_t = ng.multiply(Y_t, X_t)
    then_body_res_1 = ng.result(mmul_t)
    then_body_res_2 = ng.result(mul_t)
    then_body = GraphBody([X_t, Y_t], [then_body_res_1, then_body_res_2])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # else_body
    X_e = ng.parameter([2], np.float32, "X")
    Z_e = ng.parameter([], np.float32, "Z")
    mul_e = ng.multiply(X_e, Z_e)
    else_body_res_1 = ng.result(Z_e)
    else_body_res_2 = ng.result(mul_e)
    else_body = GraphBody([X_e, Z_e], [else_body_res_1, else_body_res_2])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(3, 1)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    X = ng.constant([3, 4], dtype=np.float32)
    Y = ng.constant([2, 1], dtype=np.float32)
    Z = ng.constant(4.0, dtype=np.float32)
    if_node = ng.if_op(condition, [X, Y, Z], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def simple_if(condition_val):
    condition = ng.constant(condition_val, dtype=np.int8)
    # then_body
    X_t = ng.parameter([2], np.float32, "X")
    Y_t = ng.parameter([2], np.float32, "Y")

    then_mul = ng.multiply(X_t, Y_t)
    then_body_res_1 = ng.result(then_mul)
    then_body = GraphBody([X_t, Y_t], [then_body_res_1])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    # else_body
    X_e = ng.parameter([2], np.float32, "X")
    Y_e = ng.parameter([2], np.float32, "Y")
    add_e = ng.add(X_e, Y_e)
    else_body_res_1 = ng.result(add_e)
    else_body = GraphBody([X_e, Y_e], [else_body_res_1])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    X = ng.constant([3, 4], dtype=np.float32)
    Y = ng.constant([2, 1], dtype=np.float32)
    if_node = ng.if_op(condition, [X, Y], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    relu = ng.relu(if_node)
    return relu


def simple_if_without_parameters(condition_val):
    condition = ng.constant(condition_val, dtype=np.bool)

    # then_body
    then_constant = ng.constant(0.7, dtype=np.float)
    then_body_res_1 = ng.result(then_constant)
    then_body = GraphBody([], [then_body_res_1])
    then_body_inputs = []
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    # else_body
    else_const = ng.constant(9.0, dtype=np.float)
    else_body_res_1 = ng.result(else_const)
    else_body = GraphBody([], [else_body_res_1])
    else_body_inputs = []
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    if_node = ng.if_op(condition, [], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    relu = ng.relu(if_node)
    return relu


def check_results(results, expected_results):
    assert len(results) == len(expected_results)
    for id_result, res in enumerate(results):
        assert np.allclose(res, expected_results[id_result])


def check_if(if_model, cond_val, exp_results):
    last_node = if_model(cond_val)
    runtime = get_runtime()
    computation = runtime.computation(last_node)
    results = computation()
    check_results(results, exp_results)


@xfail_issue_00000
def test_if_with_two_outputs():
    check_if(create_simple_if_with_two_outputs, True,
             [np.array([10], dtype=np.float32), np.array([-20], dtype=np.float32)])
    check_if(create_simple_if_with_two_outputs, False,
             [np.array([17], dtype=np.float32), np.array([16], dtype=np.float32)])


@xfail_issue_00000
def test_diff_if_with_two_outputs():
    check_if(create_diff_if_with_two_outputs, True,
             [np.array([10], dtype=np.float32), np.array([6, 4], dtype=np.float32)])
    check_if(create_diff_if_with_two_outputs, False,
             [np.array([4], dtype=np.float32), np.array([12, 16], dtype=np.float32)])


@xfail_issue_00000
def test_simple_if():
    check_if(simple_if, True, [np.array([6, 4], dtype=np.float32)])
    check_if(simple_if, False, [np.array([5, 5], dtype=np.float32)])


@xfail_issue_00000
def test_simple_if_without_body_parameters():
    check_if(simple_if_without_parameters, True, [np.array([0.7], dtype=np.float32)])
    check_if(simple_if_without_parameters, False, [np.array([9.0], dtype=np.float32)])

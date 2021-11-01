from openvino.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorSliceInputDesc,
    TensorIteratorMergedInputDesc,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
    TensorIteratorConcatOutputDesc,
)

import openvino.opset8 as ov
import numpy as np
from tests.runtime import get_runtime


def create_simple_if(condition_val):
    condition = ov.constant(condition_val, dtype=np.bool)
    # then_body
    X_t = ov.parameter([], np.float32, "X")
    Y_t = ov.parameter([], np.float32, "Y")
    Z_t = ov.parameter([], np.float32, "Z")

    add_t = ov.add(X_t, Y_t)
    mul_t = ov.multiply(Y_t, Z_t)
    then_body_res_1 = ov.result(add_t)
    then_body_res_2 = ov.result(mul_t)
    then_body = GraphBody([X_t, Y_t, Z_t], [then_body_res_1, then_body_res_2])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1),
                        TensorIteratorInvariantInputDesc(3, 2)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # else_body
    X_e = ov.parameter([], np.float32, "X")
    Z_e = ov.parameter([], np.float32, "Z")
    W_e = ov.parameter([], np.float32, "W")

    add_e = ov.add(X_e, W_e)
    pow_e = ov.power(W_e, Z_e)
    else_body_res_1 = ov.result(add_e)
    else_body_res_2 = ov.result(pow_e)
    else_body = GraphBody([X_e, Z_e, W_e], [else_body_res_1, else_body_res_2])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(3, 1),
                        TensorIteratorInvariantInputDesc(4, 2)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # condition is true

    X = ov.constant(15.0, dtype=np.float32)
    Y = ov.constant(-5.0, dtype=np.float32)
    Z = ov.constant(4.0, dtype=np.float32)
    W = ov.constant(2.0, dtype=np.float32)
    if_node = ov.if_op(condition, [X, Y, Z, W], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def create_diff_if(condition_val):
    condition = ov.constant(condition_val, dtype=np.bool)
    # then_body
    X_t = ov.parameter([2], np.float32, "X")
    Y_t = ov.parameter([2], np.float32, "Y")
    mmul_t = ov.matmul(X_t, Y_t, False, False)
    mul_t = ov.multiply(Y_t, X_t)
    then_body_res_1 = ov.result(mmul_t)
    then_body_res_2 = ov.result(mul_t)
    then_body = GraphBody([X_t, Y_t], [then_body_res_1, then_body_res_2])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # else_body
    X_e = ov.parameter([2], np.float32, "X")
    Z_e = ov.parameter([], np.float32, "Z")
    mul_e = ov.multiply(X_e, Z_e)
    else_body_res_1 = ov.result(Z_e)
    else_body_res_2 = ov.result(mul_e)
    else_body = GraphBody([X_e, Z_e], [else_body_res_1, else_body_res_2])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(3, 1)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0), TensorIteratorBodyOutputDesc(1, 1)]

    # condition is true

    X = ov.constant([3, 4], dtype=np.float32)
    Y = ov.constant([2, 1], dtype=np.float32)
    Z = ov.constant(4.0, dtype=np.float32)
    if_node = ov.if_op(condition, [X, Y, Z], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def check_results(results, expected_results):
    assert len(results) == len(expected_results)
    for id_result, res in enumerate(results):
        assert np.allclose(res, expected_results[id_result])


def check_if(if_model, cond_val, exp_results):
    if_node = if_model(cond_val)
    runtime = get_runtime()
    computation = runtime.computation(if_node)
    if_results = computation()
    check_results(if_results, exp_results)


def test_if():
    check_if(create_simple_if, True, [np.array([10], dtype=np.float32), np.array([-20], dtype=np.float32)])
    check_if(create_simple_if, False, [np.array([17], dtype=np.float32), np.array([16], dtype=np.float32)])
    check_if(create_diff_if, True, [np.array([10], dtype=np.float32),np.array([6, 4], dtype=np.float32)])
    check_if(create_diff_if, False, [np.array([4], dtype=np.float32),np.array([12, 16], dtype=np.float32)])
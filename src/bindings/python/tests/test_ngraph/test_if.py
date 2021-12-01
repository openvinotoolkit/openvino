import numpy as np
import openvino.runtime.opset8 as ov
import pytest
from openvino.runtime.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
)
from tests.runtime import get_runtime


def create_simple_if_with_two_outputs(condition_val):
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

    X = ov.constant(15.0, dtype=np.float32)
    Y = ov.constant(-5.0, dtype=np.float32)
    Z = ov.constant(4.0, dtype=np.float32)
    W = ov.constant(2.0, dtype=np.float32)
    if_node = ov.if_op(condition, [X, Y, Z, W], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def create_diff_if_with_two_outputs(condition_val):
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

    X = ov.constant([3, 4], dtype=np.float32)
    Y = ov.constant([2, 1], dtype=np.float32)
    Z = ov.constant(4.0, dtype=np.float32)
    if_node = ov.if_op(condition, [X, Y, Z], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    return if_node


def simple_if(condition_val):
    condition = ov.constant(condition_val, dtype=np.bool)
    # then_body
    X_t = ov.parameter([2], np.float32, "X")
    Y_t = ov.parameter([2], np.float32, "Y")

    then_mul = ov.multiply(X_t, Y_t)
    then_body_res_1 = ov.result(then_mul)
    then_body = GraphBody([X_t, Y_t], [then_body_res_1])
    then_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    # else_body
    X_e = ov.parameter([2], np.float32, "X")
    Y_e = ov.parameter([2], np.float32, "Y")
    add_e = ov.add(X_e, Y_e)
    else_body_res_1 = ov.result(add_e)
    else_body = GraphBody([X_e, Y_e], [else_body_res_1])
    else_body_inputs = [TensorIteratorInvariantInputDesc(1, 0), TensorIteratorInvariantInputDesc(2, 1)]
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    X = ov.constant([3, 4], dtype=np.float32)
    Y = ov.constant([2, 1], dtype=np.float32)
    if_node = ov.if_op(condition, [X, Y], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    relu = ov.relu(if_node)
    return relu


def simple_if_without_parameters(condition_val):
    condition = ov.constant(condition_val, dtype=np.bool)

    # then_body
    then_constant = ov.constant(0.7, dtype=np.float)
    then_body_res_1 = ov.result(then_constant)
    then_body = GraphBody([], [then_body_res_1])
    then_body_inputs = []
    then_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    # else_body
    else_const = ov.constant(9.0, dtype=np.float)
    else_body_res_1 = ov.result(else_const)
    else_body = GraphBody([], [else_body_res_1])
    else_body_inputs = []
    else_body_outputs = [TensorIteratorBodyOutputDesc(0, 0)]

    if_node = ov.if_op(condition, [], (then_body, else_body), (then_body_inputs, else_body_inputs),
                       (then_body_outputs, else_body_outputs))
    relu = ov.relu(if_node)
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


# After deleting evalute method for if, constant folding stopped working.
# As result bug with id 67255 began to appear
@pytest.mark.xfail(reason="bug 67255")
def test_if_with_two_outputs():
    check_if(create_simple_if_with_two_outputs, True,
             [np.array([10], dtype=np.float32), np.array([-20], dtype=np.float32)])
    check_if(create_simple_if_with_two_outputs, False,
             [np.array([17], dtype=np.float32), np.array([16], dtype=np.float32)])


@pytest.mark.xfail(reason="bug 67255")
def test_diff_if_with_two_outputs():
    check_if(create_diff_if_with_two_outputs, True,
             [np.array([10], dtype=np.float32), np.array([6, 4], dtype=np.float32)])
    check_if(create_diff_if_with_two_outputs, False,
             [np.array([4], dtype=np.float32), np.array([12, 16], dtype=np.float32)])


def test_simple_if():
    check_if(simple_if, True, [np.array([6, 4], dtype=np.float32)])
    check_if(simple_if, False, [np.array([5, 5], dtype=np.float32)])


def test_simple_if_without_body_parameters():
    check_if(simple_if_without_parameters, True, [np.array([0.7], dtype=np.float32)])
    check_if(simple_if_without_parameters, False, [np.array([9.0], dtype=np.float32)])

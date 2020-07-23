# ******************************************************************************
# Copyright 2018-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import numpy as np
import onnx
import pytest

from tests.test_onnx.utils import run_node


def import_and_compute(op_type, input_data, **node_attrs):
    data_inputs = [np.array(input_data)]
    node = onnx.helper.make_node(op_type, inputs=["x"], outputs=["y"], **node_attrs)
    return run_node(node, data_inputs).pop()


def test_reduce_max():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceMax", data, keepdims=0), np.max(data, keepdims=False))
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0,), keepdims=0), np.max(data, keepdims=False, axis=(0,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(1,), keepdims=0), np.max(data, keepdims=False, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(2,), keepdims=0), np.max(data, keepdims=False, axis=(2,))
    )

    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 1), keepdims=0),
        np.max(data, keepdims=False, axis=(0, 1)),
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 2), keepdims=0),
        np.max(data, keepdims=False, axis=(0, 2)),
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(1, 2), keepdims=0),
        np.max(data, keepdims=False, axis=(1, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 1, 2), keepdims=0),
        np.max(data, keepdims=False, axis=(0, 1, 2)),
    )


def test_reduce_max_keepdims():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceMax", data), np.max(data, keepdims=True))
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0,)), np.max(data, keepdims=True, axis=(0,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(1,)), np.max(data, keepdims=True, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(2,)), np.max(data, keepdims=True, axis=(2,))
    )

    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 1)), np.max(data, keepdims=True, axis=(0, 1))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 2)), np.max(data, keepdims=True, axis=(0, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(1, 2)), np.max(data, keepdims=True, axis=(1, 2))
    )

    assert np.array_equal(
        import_and_compute("ReduceMax", data, axes=(0, 1, 2)), np.max(data, keepdims=True, axis=(0, 1, 2))
    )


def test_reduce_min():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceMin", data), np.min(data, keepdims=True))
    assert np.array_equal(import_and_compute("ReduceMin", data, keepdims=0), np.min(data, keepdims=False))

    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(1,)), np.min(data, keepdims=True, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(1,), keepdims=0), np.min(data, keepdims=False, axis=(1,))
    )

    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(0, 2)), np.min(data, keepdims=True, axis=(0, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(0, 2), keepdims=0),
        np.min(data, keepdims=False, axis=(0, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(0, 1, 2)), np.min(data, keepdims=True, axis=(0, 1, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceMin", data, axes=(0, 1, 2), keepdims=0),
        np.min(data, keepdims=False, axis=(0, 1, 2)),
    )


def test_reduce_mean():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceMean", data), np.mean(data, keepdims=True))
    assert np.array_equal(import_and_compute("ReduceMean", data, keepdims=0), np.mean(data, keepdims=False))

    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(1,)), np.mean(data, keepdims=True, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(1,), keepdims=0),
        np.mean(data, keepdims=False, axis=(1,)),
    )

    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(0, 2)), np.mean(data, keepdims=True, axis=(0, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(0, 2), keepdims=0),
        np.mean(data, keepdims=False, axis=(0, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(0, 1, 2)), np.mean(data, keepdims=True, axis=(0, 1, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceMean", data, axes=(0, 1, 2), keepdims=0),
        np.mean(data, keepdims=False, axis=(0, 1, 2)),
    )


def test_reduce_sum():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceSum", data, keepdims=0), np.sum(data, keepdims=False))

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(1,), keepdims=0), np.sum(data, keepdims=False, axis=(1,))
    )

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(0, 2), keepdims=0),
        np.sum(data, keepdims=False, axis=(0, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(0, 1, 2), keepdims=0),
        np.sum(data, keepdims=False, axis=(0, 1, 2)),
    )


def test_reduce_sum_keepdims():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceSum", data), np.sum(data, keepdims=True))

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(1,)), np.sum(data, keepdims=True, axis=(1,))
    )

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(0, 2)), np.sum(data, keepdims=True, axis=(0, 2))
    )

    assert np.array_equal(
        import_and_compute("ReduceSum", data, axes=(0, 1, 2)), np.sum(data, keepdims=True, axis=(0, 1, 2))
    )


def test_reduce_prod():
    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceProd", data), np.prod(data, keepdims=True))
    assert np.array_equal(import_and_compute("ReduceProd", data, keepdims=0), np.prod(data, keepdims=False))

    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(1,)), np.prod(data, keepdims=True, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(1,), keepdims=0),
        np.prod(data, keepdims=False, axis=(1,)),
    )

    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(0, 2)), np.prod(data, keepdims=True, axis=(0, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(0, 2), keepdims=0),
        np.prod(data, keepdims=False, axis=(0, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(0, 1, 2)), np.prod(data, keepdims=True, axis=(0, 1, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceProd", data, axes=(0, 1, 2), keepdims=0),
        np.prod(data, keepdims=False, axis=(0, 1, 2)),
    )


@pytest.mark.parametrize("reduction_axes", [(0,), (0, 2), (0, 1, 2)])
def test_reduce_l1(reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sum(np.abs(input_data), keepdims=True, axis=reduction_axes)
    node = onnx.helper.make_node("ReduceL1", inputs=["x"], outputs=["y"], axes=reduction_axes)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sum(np.abs(input_data), keepdims=False, axis=reduction_axes)
    node = onnx.helper.make_node("ReduceL1", inputs=["x"], outputs=["y"], keepdims=0, axes=reduction_axes)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_l1_default_axes():
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sum(np.abs(input_data), keepdims=True)
    node = onnx.helper.make_node("ReduceL1", inputs=["x"], outputs=["y"])
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sum(np.abs(input_data), keepdims=False)
    node = onnx.helper.make_node("ReduceL1", inputs=["x"], outputs=["y"], keepdims=0)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


@pytest.mark.parametrize("reduction_axes", [(0,), (0, 2), (0, 1, 2)])
def test_reduce_l2(reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sqrt(np.sum(np.square(input_data), keepdims=True, axis=reduction_axes))
    node = onnx.helper.make_node("ReduceL2", inputs=["x"], outputs=["y"], axes=reduction_axes)
    raw_result = run_node(node, [input_data])
    ng_result = np.array(raw_result.pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sqrt(np.sum(np.square(input_data), keepdims=False, axis=reduction_axes))
    node = onnx.helper.make_node("ReduceL2", inputs=["x"], outputs=["y"], keepdims=0, axes=reduction_axes)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_l2_default_axes():
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sqrt(np.sum(np.square(input_data), keepdims=True))
    node = onnx.helper.make_node("ReduceL2", inputs=["x"], outputs=["y"])
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sqrt(np.sum(np.square(input_data), keepdims=False))
    node = onnx.helper.make_node("ReduceL2", inputs=["x"], outputs=["y"], keepdims=0)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


@pytest.mark.parametrize("reduction_axes", [(0,), (0, 2), (0, 1, 2)])
def test_reduce_log_sum(reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(0, 1, shape).astype(np.float32)

    expected = np.log(np.sum(input_data, keepdims=True, axis=reduction_axes))
    node = onnx.helper.make_node("ReduceLogSum", inputs=["x"], outputs=["y"], axes=reduction_axes)
    ng_result = run_node(node, [input_data]).pop()
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.log(np.sum(input_data, keepdims=False, axis=reduction_axes))
    node = onnx.helper.make_node("ReduceLogSum", inputs=["x"], outputs=["y"], keepdims=0, axes=reduction_axes)
    ng_result = run_node(node, [input_data]).pop()
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_log_sum_default_axes():
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(0, 1, shape).astype(np.float32)

    expected = np.log(np.sum(input_data, keepdims=True))
    node = onnx.helper.make_node("ReduceLogSum", inputs=["x"], outputs=["y"])
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.log(np.sum(input_data, keepdims=False))
    node = onnx.helper.make_node("ReduceLogSum", inputs=["x"], outputs=["y"], keepdims=0)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_log_sum_exp():
    def logsumexp(data, axis=None, keepdims=True):
        return np.log(np.sum(np.exp(data), axis=axis, keepdims=keepdims))

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ReduceLogSumExp", data), logsumexp(data, keepdims=True))
    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, keepdims=0), logsumexp(data, keepdims=False)
    )

    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(1,)), logsumexp(data, keepdims=True, axis=(1,))
    )
    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(1,), keepdims=0),
        logsumexp(data, keepdims=False, axis=(1,)),
    )

    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(0, 2)), logsumexp(data, keepdims=True, axis=(0, 2))
    )
    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(0, 2), keepdims=0),
        logsumexp(data, keepdims=False, axis=(0, 2)),
    )

    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(0, 1, 2)),
        logsumexp(data, keepdims=True, axis=(0, 1, 2)),
    )
    assert np.array_equal(
        import_and_compute("ReduceLogSumExp", data, axes=(0, 1, 2), keepdims=0),
        logsumexp(data, keepdims=False, axis=(0, 1, 2)),
    )


@pytest.mark.parametrize("reduction_axes", [(0,), (0, 2), (0, 1, 2)])
def test_reduce_sum_square(reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sum(np.square(input_data), keepdims=True, axis=reduction_axes)
    node = onnx.helper.make_node("ReduceSumSquare", inputs=["x"], outputs=["y"], axes=reduction_axes)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sum(np.square(input_data), keepdims=False, axis=reduction_axes)
    node = onnx.helper.make_node(
        "ReduceSumSquare", inputs=["x"], outputs=["y"], keepdims=0, axes=reduction_axes
    )
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_sum_square_default_axes():
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.uniform(-100, 100, shape).astype(np.float32)

    expected = np.sum(np.square(input_data), keepdims=True)
    node = onnx.helper.make_node("ReduceSumSquare", inputs=["x"], outputs=["y"])
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)

    expected = np.sum(np.square(input_data), keepdims=False)
    node = onnx.helper.make_node("ReduceSumSquare", inputs=["x"], outputs=["y"], keepdims=0)
    ng_result = np.array(run_node(node, [input_data]).pop())
    assert np.array_equal(expected.shape, ng_result.shape)
    assert np.allclose(expected, ng_result)


def test_reduce_argmin():
    def argmin(ndarray, axis, keepdims=False):
        res = np.argmin(ndarray, axis=axis)
        if keepdims:
            res = np.expand_dims(res, axis=axis)
        return res

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ArgMin", data, axis=0), argmin(data, keepdims=True, axis=0))
    assert np.array_equal(
        import_and_compute("ArgMin", data, axis=0, keepdims=0), argmin(data, keepdims=False, axis=0)
    )
    assert np.array_equal(import_and_compute("ArgMin", data, axis=1), argmin(data, keepdims=True, axis=1))
    assert np.array_equal(
        import_and_compute("ArgMin", data, axis=1, keepdims=0), argmin(data, keepdims=False, axis=1)
    )
    assert np.array_equal(import_and_compute("ArgMin", data, axis=2), argmin(data, keepdims=True, axis=2))
    assert np.array_equal(
        import_and_compute("ArgMin", data, axis=2, keepdims=0), argmin(data, keepdims=False, axis=2)
    )


def test_reduce_argmax():
    def argmax(ndarray, axis, keepdims=False):
        res = np.argmax(ndarray, axis=axis)
        if keepdims:
            res = np.expand_dims(res, axis=axis)
        return res

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)

    assert np.array_equal(import_and_compute("ArgMax", data, axis=0), argmax(data, keepdims=True, axis=0))
    assert np.array_equal(
        import_and_compute("ArgMax", data, axis=0, keepdims=0), argmax(data, keepdims=False, axis=0)
    )
    assert np.array_equal(import_and_compute("ArgMax", data, axis=1), argmax(data, keepdims=True, axis=1))
    assert np.array_equal(
        import_and_compute("ArgMax", data, axis=1, keepdims=0), argmax(data, keepdims=False, axis=1)
    )
    assert np.array_equal(import_and_compute("ArgMax", data, axis=2), argmax(data, keepdims=True, axis=2))
    assert np.array_equal(
        import_and_compute("ArgMax", data, axis=2, keepdims=0), argmax(data, keepdims=False, axis=2)
    )

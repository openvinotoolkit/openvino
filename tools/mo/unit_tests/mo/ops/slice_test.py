# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, dynamic_dimension_value, shape_array, \
    strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.slice import Slice, OvSlice
from unit_tests.utils.graph import build_graph, valued_const_with_data, valued_data, regular_op_with_empty_data, \
    connect, shaped_data, shaped_const_with_data


class TestSliceOp():
    @pytest.mark.parametrize("inp_value, inp_shape, starts, ends, axes, steps, expected_value, expected_shape",[
        # standard case
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, 2], [0, 1], [1, 1],
         [[5], [3], [6]], [3, 1]),
        # negative bounds
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [0, 1], [1, 1],
         [[5], [3], [6]], [3, 1]),
        # unusual order of axes
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [1, 0], [1, 1],
         [[2, 3, 5]], [1, 3]),
        # when only input_shape is defined without values (one from bottom element is shape)
        (None, [4, 5, 6], [1, 2], [4, 3], [0, 1], [1, 1], None, [3, 1, 6]),
        # boundary case
        (None, [4, 5, 6], [0, 2], [np.iinfo(np.int32).max, 3], [0, 1], [1, 1], None, [4, 1, 6]),
        # boundary case
        (None, [4, 5, 6], [np.iinfo(np.int32).min, 2], [3, 3], [0, 1], [1, 1], None, [3, 1, 6],),
        # 1D input
        ([1, 3, 224, 224], [4], [1], [2], [0], [1], [3], [1]),
        # 1D input with negative starts
        (None, [4], [-1], [1], [0], [-1], None, [2]),
        # 1D input with negative ends
        (None, [4], [1], [-1], [0], [1], None, [2]),
        # with rounding (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
        (None, [4], [1], [3], [0], [4], None, [1]),
        # with rounding and negative steps (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
        (None, [10], [7], [3], [0], [-7], None, [1]),
        # reversing the sequence of elements
        (None, [10], [-1], [np.iinfo(np.int32).min], [0], [-1], None, [10]),
        # dynamic dimensions cases
        # starts are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], None, [3, 2], [0, 1], [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # ends are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], None, [0, 1], [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # axes are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], None, [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # steps are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [0, 1], None, None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # negative steps and since after normalization starts < ends output shape has 0-size dimension
        (None, [20], [1], [-1], [0], [-2], None, [0]),
        # since starts == ends output shape has 0-size dimension
        (None, [4], [1], [1], [0], [1], None, [0]),
        # since starts > ends output shape has 0-size dimension
        (None, [4], [2], [1], [0], [1], None, [0])
    ])
    def test_slice_infer(self, inp_value, inp_shape, starts, ends, axes, steps, expected_value, expected_shape):
        if inp_value is None:
            input_node = shaped_data('data_1', int64_array(inp_shape))
        else:
            input_node = valued_data('data_1', int64_array(inp_value))
        if inp_value is not None and inp_shape is not None:
            assert np.array_equal(np.array(inp_value).shape, inp_shape)

        def convert_args(val, name=''):
            if val is not None:
                return valued_const_with_data(name, int64_array(val))
            else:
                return shaped_const_with_data(name, [0])  # fake shape

        starts = convert_args(starts, 'starts')
        ends = convert_args(ends, 'ends')
        axes = convert_args(axes, 'axes')
        steps = convert_args(steps, 'steps')
        if expected_shape is not None:
            expected_shape = shape_array(expected_shape)

        nodes = {
            **input_node,
            **regular_op_with_empty_data('slice', {'op': 'Slice'}),
            **starts,
            **ends,
            **axes,
            **steps,
        }

        graph = build_graph(nodes,
                            [('data_1', 'slice'),
                             *connect('starts', '1:slice'),
                             *connect('ends', '2:slice'),
                             *connect('axes', '3:slice'),
                             *connect('steps', '4:slice'),
                             *connect('slice', 'slice_d')])

        graph.stage = 'middle'
        slice_node = Node(graph, 'slice')

        Slice.infer(slice_node)
        if expected_value is not None:
            assert strict_compare_tensors(slice_node.out_node().value, expected_value)
        assert strict_compare_tensors(slice_node.out_node().shape, expected_shape)


class TestOvSliceOp():
    @pytest.mark.parametrize("inp_value, inp_shape, starts, ends, axes, steps, expected_value, expected_shape",[
        # standard case
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, 2], [0, 1], [1, 1],
         [[5], [3], [6]], [3, 1]),
        # negative bounds
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [0, 1], [1, 1],
         [[5], [3], [6]], [3, 1]),
        # unusual order of axes
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [1, 0], [1, 1],
         [[2, 3, 5]], [1, 3]),
        # when only input_shape is defined without values (one from bottom element is shape)
        (None, [4, 5, 6], [1, 2], [4, 3], [0, 1], [1, 1], None, [3, 1, 6]),
        # boundary case
        (None, [4, 5, 6], [0, 2], [np.iinfo(np.int32).max, 3], [0, 1], [1, 1], None, [4, 1, 6]),
        # boundary case
        (None, [4, 5, 6], [np.iinfo(np.int32).min, 2], [3, 3], [0, 1], [1, 1], None, [3, 1, 6],),
        # 1D input
        ([1, 3, 224, 224], [4], [1], [2], [0], [1], [3], [1]),
        # 1D input with negative starts
        (None, [4], [-1], [1], [0], [-1], None, [2]),
        # 1D input with negative ends
        (None, [4], [1], [-1], [0], [1], None, [2]),
        # with rounding (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
        (None, [4], [1], [3], [0], [4], None, [1]),
        # with rounding and negative steps (e.g. take from 1st to 3rd with step 4 should give shape 1 not 0)
        (None, [10], [7], [3], [0], [-7], None, [1]),
        # reversing the sequence of elements
        (None, [10], [-1], [np.iinfo(np.int32).min], [0], [-1], None, [10]),
        # dynamic dimensions cases
        # starts are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], None, [3, 2], [0, 1], [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # ends are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], None, [0, 1], [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # axes are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], None, [1, 1], None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # steps are non-constant
        ([[4, 5, 6, 7], [2, 3, 5, 6], [5, 6, 8, 9], [5, 6, 8, 9]], [4, 4], [0, 1], [3, -2], [0, 1], None, None,
         [dynamic_dimension_value, dynamic_dimension_value]),
        # negative steps and since after normalization starts < ends output shape has 0-size dimension
        (None, [20], [1], [-1], [0], [-2], None, [0]),
        # since starts == ends output shape has 0-size dimension
        (None, [4], [1], [1], [0], [1], None, [0]),
        # since starts > ends output shape has 0-size dimension
        (None, [4], [2], [1], [0], [1], None, [0])
    ])
    def test_ov_slice_infer(self, inp_value, inp_shape, starts, ends, axes, steps, expected_value, expected_shape):
        if inp_value is None:
            input_node = shaped_data('data_1', int64_array(inp_shape))
        else:
            input_node = valued_data('data_1', int64_array(inp_value))
        if inp_value is not None and inp_shape is not None:
            assert np.array_equal(np.array(inp_value).shape, inp_shape)

        def convert_args(val, name=''):
            if val is not None:
                return valued_const_with_data(name, int64_array(val))
            else:
                return shaped_const_with_data(name, [0])  # fake shape

        starts = convert_args(starts, 'starts')
        ends = convert_args(ends, 'ends')
        steps = convert_args(steps, 'steps')
        axes = convert_args(axes, 'axes')
        if expected_shape is not None:
            expected_shape = shape_array(expected_shape)

        nodes = {
            **input_node,
            **regular_op_with_empty_data('slice', {'op': 'OvSlice'}),
            **starts,
            **ends,
            **steps,
            **axes,
        }

        graph = build_graph(nodes,
                            [('data_1', 'slice'),
                             *connect('starts', '1:slice'),
                             *connect('ends', '2:slice'),
                             *connect('steps', '3:slice'),
                             *connect('axes', '4:slice'),
                             *connect('slice', 'slice_d')])

        graph.stage = 'middle'
        slice_node = Node(graph, 'slice')

        OvSlice.infer(slice_node)
        if expected_value is not None:
            assert strict_compare_tensors(slice_node.out_node().value, expected_value)
        assert strict_compare_tensors(slice_node.out_node().shape, expected_shape)

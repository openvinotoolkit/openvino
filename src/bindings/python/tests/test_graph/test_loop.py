# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import openvino.runtime.opset8 as ov
from openvino import Model, Shape

from openvino.runtime.op.util import (
    InvariantInputDescription,
    BodyOutputDescription,
    SliceInputDescription,
    MergedInputDescription,
    ConcatOutputDescription,
)
from tests.utils.helpers import compare_models


def test_simple_loop():
    param_x = ov.parameter(Shape([32, 1, 10]), np.float32, "X")
    param_y = ov.parameter(Shape([32, 1, 10]), np.float32, "Y")
    param_m = ov.parameter(Shape([32, 1, 10]), np.float32, "M")

    input_shape = Shape([])

    current_iteration = ov.parameter(Shape([1]), np.int64)
    x_i = ov.parameter(input_shape, np.float32)
    y_i = ov.parameter(input_shape, np.float32)
    m_body = ov.parameter(input_shape, np.float32)
    bool_val = np.array([1], dtype=bool)
    bool_val[0] = True
    body_condition = ov.constant(bool_val)
    trip_count = ov.constant(10, dtype=np.int64)
    exec_condition = ov.constant(True, dtype=bool)

    add = ov.add(x_i, y_i)
    zo = ov.multiply(add, m_body)

    body = Model([body_condition, zo], [current_iteration, x_i, y_i, m_body], "body_function")

    loop = ov.loop(trip_count, exec_condition)
    loop.set_function(body)
    loop.set_invariant_input(x_i, param_x.output(0))
    loop.set_invariant_input(y_i, param_y.output(0))
    loop.set_merged_input(m_body, param_m.output(0), zo.output(0))
    loop.set_special_body_ports([-1, 0])

    out0 = loop.get_iter_value(body_condition.output(0), -1)
    out1 = loop.get_iter_value(zo.output(0), -1)
    out2 = loop.get_concatenated_slices(zo.output(0), 0, 1, 1, -1, 1)

    result0 = ov.result(out0)
    result1 = ov.result(out1)
    result2 = ov.result(out2)

    out0_shape = [1]
    out1_shape = [32, 1, 10]
    out2_shape = [32, 10, 10]

    assert list(result0.get_output_shape(0)) == out0_shape
    assert list(result1.get_output_shape(0)) == out1_shape
    assert list(result2.get_output_shape(0)) == out2_shape

    assert list(loop.get_output_shape(0)) == out0_shape
    assert list(loop.get_output_shape(1)) == out1_shape
    assert list(loop.get_output_shape(2)) == out2_shape


def test_loop_inputs_are_nodes():
    param_x = ov.parameter(Shape([32, 1, 10]), np.float32, "X")
    param_y = ov.parameter(Shape([32, 1, 10]), np.float32, "Y")
    param_m = ov.parameter(Shape([32, 1, 10]), np.float32, "M")

    input_shape = Shape([])

    current_iteration = ov.parameter(Shape([1]), np.int64)
    x_i = ov.parameter(input_shape, np.float32)
    y_i = ov.parameter(input_shape, np.float32)
    m_body = ov.parameter(input_shape, np.float32)
    bool_val = np.array([1], dtype=bool)
    bool_val[0] = True
    body_condition = ov.constant(bool_val)
    trip_shape = ov.parameter([10], np.int64, "trip_shapeof")
    trip_count = ov.shape_of(trip_shape)
    exp_shape_size = ov.constant(10, np.int64)
    exec_condition = ov.equal(exp_shape_size, trip_count)

    add = ov.add(x_i, y_i)
    zo = ov.multiply(add, m_body)

    body = Model([body_condition, zo], [current_iteration, x_i, y_i, m_body], "body_function")

    loop = ov.loop(trip_count, exec_condition)
    loop.set_function(body)
    loop.set_invariant_input(x_i, param_x.output(0))
    loop.set_invariant_input(y_i, param_y.output(0))
    loop.set_merged_input(m_body, param_m.output(0), zo.output(0))
    loop.set_special_body_ports([-1, 0])

    loop.constructor_validate_and_infer_types()

    out0 = loop.get_iter_value(body_condition.output(0), -1)
    out1 = loop.get_iter_value(zo.output(0), -1)
    out2 = loop.get_concatenated_slices(zo.output(0), 0, 1, 1, -1, 1)

    result0 = ov.result(out0)
    result1 = ov.result(out1)
    result2 = ov.result(out2)

    out0_shape = [1]
    out1_shape = [32, 1, 10]
    out2_shape = [32, 10, 10]

    assert list(result0.get_output_shape(0)) == out0_shape
    assert list(result1.get_output_shape(0)) == out1_shape
    assert list(result2.get_output_shape(0)) == out2_shape

    assert list(loop.get_output_shape(0)) == out0_shape
    assert list(loop.get_output_shape(1)) == out1_shape
    assert list(loop.get_output_shape(2)) == out2_shape


def test_loop_basic():
    bool_val = np.array([1], dtype=bool)
    bool_val[0] = True
    condition = ov.constant(bool_val)
    trip_count = ov.constant(16, dtype=np.int32)
    #  Body parameters
    body_timestep = ov.parameter([], np.int32, "timestep")
    body_data_in = ov.parameter([1, 2, 2], np.float32, "body_in")
    body_prev_cma = ov.parameter([2, 2], np.float32, "body_prev_cma")
    body_const_one = ov.parameter([], np.int32, "body_const_one")
    body_ports = [-1, 2]

    # CMA = cumulative moving average
    prev_cum_sum = ov.multiply(ov.convert(body_timestep, "f32"), body_prev_cma)
    curr_cum_sum = ov.add(prev_cum_sum, ov.squeeze(body_data_in, [0]))
    elem_cnt = ov.add(body_const_one, body_timestep)
    curr_cma = ov.divide(curr_cum_sum, ov.convert(elem_cnt, "f32"))
    cma_hist = ov.unsqueeze(curr_cma, [0])

    # TI inputs
    data = ov.parameter([16, 2, 2], np.float32, "data")
    # Iterations count
    zero = ov.constant(0, dtype=np.int32)
    one = ov.constant(1, dtype=np.int32)
    initial_cma = ov.constant(np.zeros([2, 2], dtype=np.float32), dtype=np.float32)
    iter_cnt = ov.range(zero, np.int32(16), np.int32(1))
    body_const_condition = ov.constant(bool_val)

    graph_body = Model(
        [curr_cma, cma_hist, body_const_condition],
        [body_timestep, body_data_in, body_prev_cma, body_const_one],
        "body_function",
    )

    ti_slice_input_desc = [
        # timestep
        # input_idx, body_param_idx, start, stride, part_size, end, axis
        SliceInputDescription(2, 0, 0, 1, 1, -1, 0),
        # data
        SliceInputDescription(3, 1, 0, 1, 1, -1, 0),
    ]
    ti_merged_input_desc = [
        # body prev/curr_cma
        MergedInputDescription(4, 2, 0),
    ]
    ti_invariant_input_desc = [
        # body const one
        InvariantInputDescription(5, 3),
    ]

    # TI outputs
    ti_body_output_desc = [
        # final average
        BodyOutputDescription(0, 0, -1),
    ]
    ti_concat_output_desc = [
        # history of cma
        ConcatOutputDescription(1, 1, 0, 1, 1, -1, 0),
    ]

    loop = ov.loop(trip_count, condition)
    loop.set_function(graph_body)
    loop.set_special_body_ports(body_ports)
    loop.set_sliced_input(body_timestep, iter_cnt.output(0), 0, 1, 1, 0, 0)
    loop.set_sliced_input(body_data_in, data.output(0), 0, 1, 1, 0, 0)

    # set different end parameter
    loop.set_input_descriptions(ti_slice_input_desc)
    loop.set_merged_input(body_prev_cma, initial_cma.output(0), curr_cma.output(0))
    loop.set_invariant_input(body_const_one, one.output(0))

    loop.get_iter_value(curr_cma.output(0), -1)
    loop.get_concatenated_slices(cma_hist.output(0), 0, 1, 1, -1, 0)

    subgraph_func = loop.get_function()

    assert isinstance(subgraph_func, type(graph_body))
    assert subgraph_func._get_raw_address() == graph_body._get_raw_address()
    assert compare_models(subgraph_func, graph_body)
    assert loop.get_special_body_ports() == body_ports
    assert loop.get_num_iterations() == 16

    input_desc = loop.get_input_descriptions()
    output_desc = loop.get_output_descriptions()

    assert len(input_desc) == len(ti_slice_input_desc) + len(
        ti_merged_input_desc,
    ) + len(ti_invariant_input_desc)
    assert len(output_desc) == len(ti_body_output_desc) + len(ti_concat_output_desc)

    for i in range(len(ti_slice_input_desc)):
        assert input_desc[i].get_type_info() == ti_slice_input_desc[i].get_type_info()
        assert input_desc[i].input_index == ti_slice_input_desc[i].input_index
        assert (
            input_desc[i].body_parameter_index
            == ti_slice_input_desc[i].body_parameter_index
        )
        assert input_desc[i].start == ti_slice_input_desc[i].start
        assert input_desc[i].stride == ti_slice_input_desc[i].stride
        assert input_desc[i].part_size == ti_slice_input_desc[i].part_size
        assert input_desc[i].end == ti_slice_input_desc[i].end
        assert input_desc[i].axis == ti_slice_input_desc[i].axis

    for i in range(len(ti_merged_input_desc)):
        assert (
            input_desc[len(ti_slice_input_desc) + i].get_type_info()
            == ti_merged_input_desc[i].get_type_info()
        )
        assert (
            input_desc[len(ti_slice_input_desc) + i].input_index
            == ti_merged_input_desc[i].input_index
        )
        assert (
            input_desc[len(ti_slice_input_desc) + i].body_parameter_index
            == ti_merged_input_desc[i].body_parameter_index
        )
        assert (
            input_desc[len(ti_slice_input_desc) + i].body_value_index
            == ti_merged_input_desc[i].body_value_index
        )

    for i in range(len(ti_concat_output_desc)):
        assert (
            output_desc[len(ti_body_output_desc) + i].get_type_info()
            == ti_concat_output_desc[i].get_type_info()
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].output_index
            == ti_concat_output_desc[i].output_index
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].body_value_index
            == ti_concat_output_desc[i].body_value_index
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].start
            == ti_concat_output_desc[i].start
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].stride
            == ti_concat_output_desc[i].stride
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].part_size
            == ti_concat_output_desc[i].part_size
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].end
            == ti_concat_output_desc[i].end
        )
        assert (
            output_desc[len(ti_body_output_desc) + i].axis
            == ti_concat_output_desc[i].axis
        )

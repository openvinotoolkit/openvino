# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.opset8 as ov
from openvino import Model, Shape

from openvino.op.util import (
    InvariantInputDescription,
    BodyOutputDescription,
    SliceInputDescription,
    MergedInputDescription,
    ConcatOutputDescription,
)
from tests.utils.helpers import compare_models


def test_simple_tensor_iterator():
    param_x = ov.parameter(Shape([32, 40, 10]), np.float32, "X")
    param_y = ov.parameter(Shape([32, 40, 10]), np.float32, "Y")
    param_m = ov.parameter(Shape([32, 2, 10]), np.float32, "M")

    x_i = ov.parameter(Shape([32, 2, 10]), np.float32, "X_i")
    y_i = ov.parameter(Shape([32, 2, 10]), np.float32, "Y_i")
    m_body = ov.parameter(Shape([32, 2, 10]), np.float32, "M_body")

    add = ov.add(x_i, y_i)
    zo = ov.multiply(add, m_body)

    body = Model([zo], [x_i, y_i, m_body], "body_function")

    ti = ov.tensor_iterator()
    ti.set_body(body)
    ti.set_sliced_input(x_i, param_x.output(0), 0, 2, 2, 39, 1)
    ti.set_sliced_input(y_i, param_y.output(0), 0, 2, 2, -1, 1)
    ti.set_invariant_input(m_body, param_m.output(0))

    out0 = ti.get_iter_value(zo.output(0), -1)
    out1 = ti.get_concatenated_slices(zo.output(0), 0, 2, 2, 39, 1)

    result0 = ov.result(out0)
    result1 = ov.result(out1)

    out0_shape = [32, 2, 10]
    out1_shape = [32, 40, 10]

    assert list(result0.get_output_shape(0)) == out0_shape
    assert list(result1.get_output_shape(0)) == out1_shape

    assert list(ti.get_output_shape(0)) == out0_shape
    assert list(ti.get_output_shape(1)) == out1_shape


def test_tensor_iterator_basic():
    #  Body parameters
    body_timestep = ov.parameter([], np.int32, "timestep")
    body_data_in = ov.parameter([1, 2, 2], np.float32, "body_in")
    body_prev_cma = ov.parameter([2, 2], np.float32, "body_prev_cma")
    body_const_one = ov.parameter([], np.int32, "body_const_one")

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

    graph_body = Model(
        [curr_cma, cma_hist],
        [body_timestep, body_data_in, body_prev_cma, body_const_one],
        "body_function",
    )

    ti_slice_input_desc = [
        # timestep
        # input_idx, body_param_idx, start, stride, part_size, end, axis
        SliceInputDescription(0, 0, 0, 1, 1, -1, 0),
        # data
        SliceInputDescription(1, 1, 0, 1, 1, -1, 0),
    ]
    ti_merged_input_desc = [
        # body prev/curr_cma
        MergedInputDescription(2, 2, 0),
    ]
    ti_invariant_input_desc = [
        # body const one
        InvariantInputDescription(3, 3),
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

    ti = ov.tensor_iterator()
    ti.set_function(graph_body)
    ti.set_sliced_input(body_timestep, iter_cnt.output(0), 0, 1, 1, 0, 0)
    ti.set_sliced_input(body_data_in, data.output(0), 0, 1, 1, 0, 0)

    # set different end parameter
    ti.set_input_descriptions(ti_slice_input_desc)
    ti.set_merged_input(body_prev_cma, initial_cma.output(0), curr_cma.output(0))
    ti.set_invariant_input(body_const_one, one.output(0))

    ti.get_iter_value(curr_cma.output(0), -1)
    ti.get_concatenated_slices(cma_hist.output(0), 0, 1, 1, -1, 0)

    subgraph_func = ti.get_function()

    assert isinstance(subgraph_func, type(graph_body))
    assert compare_models(subgraph_func, graph_body)
    assert subgraph_func._get_raw_address() == graph_body._get_raw_address()
    assert ti.get_num_iterations() == 16

    input_desc = ti.get_input_descriptions()
    output_desc = ti.get_output_descriptions()

    assert len(input_desc) == len(ti_slice_input_desc) + len(ti_merged_input_desc) + len(ti_invariant_input_desc)
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

    for i in range(len(ti_body_output_desc)):
        assert output_desc[i].get_type_info() == ti_body_output_desc[i].get_type_info()
        assert output_desc[i].output_index == ti_body_output_desc[i].output_index
        assert (
            output_desc[i].body_value_index == ti_body_output_desc[i].body_value_index
        )
        assert output_desc[i].iteration == ti_body_output_desc[i].iteration

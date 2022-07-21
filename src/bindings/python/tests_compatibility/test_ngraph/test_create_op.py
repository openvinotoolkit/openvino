# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import pytest

from _pyngraph import PartialShape, Dimension

import ngraph as ng
import ngraph.opset1 as ng_opset1
import ngraph.opset5 as ng_opset5
import ngraph.opset10 as ng_opset10
from ngraph.utils.types import make_constant_node
from ngraph.exceptions import UserInputError
from ngraph.impl import Type

np_types = [np.float32, np.int32]
integral_np_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_adaptive_avg_pool(dtype):
    data = ng.parameter([2, 24, 34, 62], name="input", dtype=dtype)
    output_shape = ng.constant(np.array([16, 16], dtype=np.int32))

    node = ng.adaptive_avg_pool(data, output_shape)

    assert node.get_type_name() == "AdaptiveAvgPool"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 24, 16, 16]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ind_type", ["i32", "i64"])
def test_adaptive_max_pool(dtype, ind_type):
    data = ng.parameter([2, 24, 34, 62], name="input", dtype=dtype)
    output_shape = ng.constant(np.array([16, 16], dtype=np.int32))

    node = ng.adaptive_max_pool(data, output_shape, ind_type)

    assert node.get_type_name() == "AdaptiveMaxPool"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [2, 24, 16, 16]
    assert list(node.get_output_shape(1)) == [2, 24, 16, 16]
    assert node.get_output_element_type(1) == Type.i32 if ind_type == "i32" else Type.i64


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_binary_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    mode = "xnor-popcount"
    pad_value = 0.0

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.binary_convolution(
        parameter_input0, parameter_input1, strides, pads_begin, pads_end, dilations, mode, pad_value,
    )

    assert node.get_type_name() == "BinaryConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_ctc_greedy_decoder(dtype):
    input0_shape = [20, 8, 128]
    input1_shape = [20, 8]
    expected_shape = [8, 20, 1, 1]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.ctc_greedy_decoder(parameter_input0, parameter_input1)

    assert node.get_type_name() == "CTCGreedyDecoder"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("fp_dtype, int_dtype, int_ci, int_sl, merge_repeated, blank_index",
                         [
                             (np.float32, np.int32, "i32", "i32", True, True),
                             (np.float32, np.int32, "i64", "i32", True, True),
                             (np.float32, np.int32, "i32", "i64", True, True),
                             (np.float32, np.int32, "i64", "i64", True, True),
                             (np.float64, np.int64, "i32", "i32", False, True),
                             (np.float64, np.int64, "i64", "i32", False, True),
                             (np.float64, np.int64, "i32", "i64", False, True),
                             (np.float64, np.int64, "i64", "i64", False, True),
                             (np.float32, np.int32, "i32", "i32", True, False),
                             (np.float32, np.int32, "i64", "i32", True, False),
                             (np.float32, np.int32, "i32", "i64", True, False),
                             (np.float32, np.int32, "i64", "i64", True, False),
                             (np.float64, np.int64, "i32", "i32", False, False),
                             (np.float64, np.int64, "i64", "i32", False, False),
                             (np.float64, np.int64, "i32", "i64", False, False),
                             (np.float64, np.int64, "i64", "i64", False, False)
                         ], )
def test_ctc_greedy_decoder_seq_len(fp_dtype, int_dtype, int_ci, int_sl, merge_repeated, blank_index):
    input0_shape = [8, 20, 128]
    input1_shape = [8]
    input2_shape = [1]
    expected_shape = [8, 20]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=fp_dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=int_dtype)
    parameter_input2 = None
    if blank_index:
        parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=int_dtype)

    node = ng.ctc_greedy_decoder_seq_len(
        parameter_input0, parameter_input1, parameter_input2, merge_repeated, int_ci, int_sl
    )

    assert node.get_type_name() == "CTCGreedyDecoderSeqLen"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_convolution_opset1(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 18, 7, 7]
    input2_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)

    node = ng_opset1.deformable_convolution(
        parameter_input0, parameter_input1, parameter_input2, strides, pads_begin, pads_end, dilations,
    )

    assert node.get_type_name() == "DeformableConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 18, 7, 7]
    input2_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)

    node = ng.deformable_convolution(
        parameter_input0, parameter_input1, parameter_input2, strides, pads_begin, pads_end, dilations,
    )

    assert node.get_type_name() == "DeformableConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_convolution_mask(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 18, 7, 7]
    input2_shape = [1, 1, 3, 3]
    input3_shape = [1, 9, 7, 7]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)
    parameter_input3 = ng.parameter(input3_shape, name="Input3", dtype=dtype)

    node = ng.deformable_convolution(
        parameter_input0, parameter_input1, parameter_input2, strides,
        pads_begin, pads_end, dilations, parameter_input3
    )

    assert node.get_type_name() == "DeformableConvolution"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_deformable_psroi_pooling(dtype):
    output_dim = 8
    spatial_scale = 0.0625
    group_size = 7
    mode = "bilinear_deformable"
    spatial_bins_x = 4
    spatial_bins_y = 4
    trans_std = 0.1
    part_size = 7

    input0_shape = [1, 392, 38, 63]
    input1_shape = [300, 5]
    input2_shape = [300, 2, 7, 7]
    expected_shape = [300, 8, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)

    node = ng.deformable_psroi_pooling(
        parameter_input0,
        parameter_input1,
        output_dim,
        spatial_scale,
        group_size,
        mode,
        spatial_bins_x,
        spatial_bins_y,
        trans_std,
        part_size,
        offsets=parameter_input2,
    )

    assert node.get_type_name() == "DeformablePSROIPooling"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize(
    ("data_shape", "rois", "batch_indices", "pooled_h", "pooled_w", "sampling_ratio", "spatial_scale", "mode", "aligned_mode", "expected_shape"),
    [
        ([2, 3, 5, 6], [7, 4], [7], 2, 2, 1, 1.0, "avg", "asymmetric", [7, 3, 2, 2]),
        ([10, 3, 5, 5], [7, 4], [7], 3, 4, 1, 1.0, "avg", "half_pixel_for_nn", [7, 3, 3, 4]),
        ([10, 3, 5, 5], [3, 4], [3], 3, 4, 1, 1.0, "avg", "half_pixel", [3, 3, 3, 4]),
        ([10, 3, 5, 5], [3, 4], [3], 3, 4, 1, np.float(1), "avg", "half_pixel", [3, 3, 3, 4]),
    ],
)
def test_roi_align(data_shape, rois, batch_indices, pooled_h, pooled_w, sampling_ratio, spatial_scale, mode, aligned_mode, expected_shape):
    data_parameter = ng.parameter(data_shape, name="Data", dtype=np.float32)
    rois_parameter = ng.parameter(rois, name="Rois", dtype=np.float32)
    batch_indices_parameter = ng.parameter(batch_indices, name="Batch_indices", dtype=np.int32)

    node = ng.roi_align(
        data_parameter,
        rois_parameter,
        batch_indices_parameter,
        pooled_h,
        pooled_w,
        sampling_ratio,
        np.float32(spatial_scale),
        mode,
        aligned_mode,
    )

    assert node.get_type_name() == "ROIAlign"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_floor_mod(dtype):
    input0_shape = [8, 1, 6, 1]
    input1_shape = [7, 1, 5]
    expected_shape = [8, 7, 6, 5]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)

    node = ng.floor_mod(parameter_input0, parameter_input1)

    assert node.get_type_name() == "FloorMod"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", np_types)
def test_gather_tree(dtype):
    input0_shape = [100, 1, 10]
    input1_shape = [100, 1, 10]
    input2_shape = [1]
    input3_shape = []
    expected_shape = [100, 1, 10]

    parameter_input0 = ng.parameter(input0_shape, name="Input0", dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name="Input1", dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name="Input2", dtype=dtype)
    parameter_input3 = ng.parameter(input3_shape, name="Input3", dtype=dtype)

    node = ng.gather_tree(parameter_input0, parameter_input1, parameter_input2, parameter_input3)

    assert node.get_type_name() == "GatherTree"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_cell_operator(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    C_t_shape = [batch_size, hidden_size]
    W_shape = [4 * hidden_size, input_size]
    R_shape = [4 * hidden_size, hidden_size]
    B_shape = [4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    expected_shape = [1, 128]

    node_default = ng.lstm_cell(
        parameter_X, parameter_H_t, parameter_C_t, parameter_W, parameter_R, parameter_B, hidden_size,
    )

    assert node_default.get_type_name() == "LSTMCell"
    assert node_default.get_output_size() == 2
    assert list(node_default.get_output_shape(0)) == expected_shape
    assert list(node_default.get_output_shape(1)) == expected_shape

    activations = ["tanh", "Sigmoid", "RELU"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 0.5

    node_param = ng.lstm_cell(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMCell"
    assert node_param.get_output_size() == 2
    assert list(node_param.get_output_shape(0)) == expected_shape
    assert list(node_param.get_output_shape(1)) == expected_shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_cell_operator_opset1(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    C_t_shape = [batch_size, hidden_size]
    W_shape = [4 * hidden_size, input_size]
    R_shape = [4 * hidden_size, hidden_size]
    B_shape = [4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    expected_shape = [1, 128]

    node_default = ng_opset1.lstm_cell(
        parameter_X, parameter_H_t, parameter_C_t, parameter_W, parameter_R, parameter_B, hidden_size,
    )

    assert node_default.get_type_name() == "LSTMCell"
    assert node_default.get_output_size() == 2
    assert list(node_default.get_output_shape(0)) == expected_shape
    assert list(node_default.get_output_shape(1)) == expected_shape

    activations = ["tanh", "Sigmoid", "RELU"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 0.5

    node_param = ng_opset1.lstm_cell(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMCell"
    assert node_param.get_output_size() == 2
    assert list(node_param.get_output_shape(0)) == expected_shape
    assert list(node_param.get_output_shape(1)) == expected_shape


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_bidirectional_opset1(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128
    num_directions = 2
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "BIDIRECTIONAL"
    node = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_reverse_opset1(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "REVERSE"

    node_default = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_forward_opset1(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "forward"

    node_default = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [2.0]
    activation_beta = [1.0]
    clip = 0.5

    node = ng_opset1.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3


def test_gru_cell_operator():
    batch_size = 1
    input_size = 16
    hidden_size = 128

    X_shape = [batch_size, input_size]
    H_t_shape = [batch_size, hidden_size]
    W_shape = [3 * hidden_size, input_size]
    R_shape = [3 * hidden_size, hidden_size]
    B_shape = [3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    expected_shape = [1, 128]

    node_default = ng.gru_cell(parameter_X, parameter_H_t, parameter_W, parameter_R, parameter_B, hidden_size)

    assert node_default.get_type_name() == "GRUCell"
    assert node_default.get_output_size() == 1
    assert list(node_default.get_output_shape(0)) == expected_shape

    activations = ["tanh", "relu"]
    activations_alpha = [1.0, 2.0]
    activations_beta = [1.0, 2.0]
    clip = 0.5
    linear_before_reset = True

    # If *linear_before_reset* is set True, then B tensor shape must be [4 * hidden_size]
    B_shape = [4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    node_param = ng.gru_cell(
        parameter_X,
        parameter_H_t,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        activations,
        activations_alpha,
        activations_beta,
        clip,
        linear_before_reset,
    )

    assert node_param.get_type_name() == "GRUCell"
    assert node_param.get_output_size() == 1
    assert list(node_param.get_output_shape(0)) == expected_shape


def test_gru_sequence():
    batch_size = 2
    input_size = 16
    hidden_size = 32
    seq_len = 8
    seq_lengths = [seq_len] * batch_size
    num_directions = 1
    direction = "FORWARD"

    X_shape = [batch_size, seq_len, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    W_shape = [num_directions, 3 * hidden_size, input_size]
    R_shape = [num_directions, 3 * hidden_size, hidden_size]
    B_shape = [num_directions, 3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    expected_shape_y = [batch_size, num_directions, seq_len, hidden_size]
    expected_shape_h = [batch_size, num_directions, hidden_size]

    node_default = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        seq_lengths,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "GRUSequence"
    assert node_default.get_output_size() == 2
    assert list(node_default.get_output_shape(0)) == expected_shape_y
    assert list(node_default.get_output_shape(1)) == expected_shape_h

    activations = ["tanh", "relu"]
    activations_alpha = [1.0, 2.0]
    activations_beta = [1.0, 2.0]
    clip = 0.5
    linear_before_reset = True

    # If *linear_before_reset* is set True, then B tensor shape must be [4 * hidden_size]
    B_shape = [num_directions, 4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    node_param = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        seq_lengths,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activations_alpha,
        activations_beta,
        clip,
        linear_before_reset,
    )

    assert node_param.get_type_name() == "GRUSequence"
    assert node_param.get_output_size() == 2
    assert list(node_param.get_output_shape(0)) == expected_shape_y
    assert list(node_param.get_output_shape(1)) == expected_shape_h


def test_rnn_sequence():
    batch_size = 2
    input_size = 16
    hidden_size = 32
    seq_len = 8
    seq_lengths = [seq_len] * batch_size
    num_directions = 1
    direction = "FORWARD"

    X_shape = [batch_size, seq_len, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    W_shape = [num_directions, hidden_size, input_size]
    R_shape = [num_directions, hidden_size, hidden_size]
    B_shape = [num_directions, hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=np.float32)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=np.float32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=np.float32)
    parameter_R = ng.parameter(R_shape, name="R", dtype=np.float32)
    parameter_B = ng.parameter(B_shape, name="B", dtype=np.float32)

    expected_shape_y = [batch_size, num_directions, seq_len, hidden_size]
    expected_shape_h = [batch_size, num_directions, hidden_size]

    node_default = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        seq_lengths,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "RNNSequence"
    assert node_default.get_output_size() == 2
    assert list(node_default.get_output_shape(0)) == expected_shape_y
    assert list(node_default.get_output_shape(1)) == expected_shape_h

    activations = ["relu"]
    activations_alpha = [2.0]
    activations_beta = [1.0]
    clip = 0.5

    node_param = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        seq_lengths,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activations_alpha,
        activations_beta,
        clip,
    )

    assert node_param.get_type_name() == "RNNSequence"
    assert node_param.get_output_size() == 2
    assert list(node_param.get_output_shape(0)) == expected_shape_y
    assert list(node_param.get_output_shape(1)) == expected_shape_h


def test_loop():
    from ngraph.utils.tensor_iterator_types import (
        GraphBody,
        TensorIteratorSliceInputDesc,
        TensorIteratorMergedInputDesc,
        TensorIteratorInvariantInputDesc,
        TensorIteratorBodyOutputDesc,
        TensorIteratorConcatOutputDesc,
    )

    condition = ng.constant(True, dtype=np.bool)
    trip_count = ng.constant(16, dtype=np.int32)
    #  Body parameters
    body_timestep = ng.parameter([], np.int32, "timestep")
    body_data_in = ng.parameter([1, 2, 2], np.float32, "body_in")
    body_prev_cma = ng.parameter([2, 2], np.float32, "body_prev_cma")
    body_const_one = ng.parameter([], np.int32, "body_const_one")

    # CMA = cumulative moving average
    prev_cum_sum = ng.multiply(ng.convert(body_timestep, "f32"), body_prev_cma)
    curr_cum_sum = ng.add(prev_cum_sum, ng.squeeze(body_data_in, [0]))
    elem_cnt = ng.add(body_const_one, body_timestep)
    curr_cma = ng.divide(curr_cum_sum, ng.convert(elem_cnt, "f32"))
    cma_hist = ng.unsqueeze(curr_cma, [0])

    # TI inputs
    data = ng.parameter([16, 2, 2], np.float32, "data")
    # Iterations count
    zero = ng.constant(0, dtype=np.int32)
    one = ng.constant(1, dtype=np.int32)
    initial_cma = ng.constant(np.zeros([2, 2], dtype=np.float32), dtype=np.float32)
    iter_cnt = ng.range(zero, np.int32(16), np.int32(1))
    ti_inputs = [iter_cnt, data, initial_cma, one]
    body_const_condition = ng.constant(True, dtype=np.bool)

    graph_body = GraphBody([body_timestep, body_data_in, body_prev_cma, body_const_one],
                           [curr_cma, cma_hist, body_const_condition])
    ti_slice_input_desc = [
        # timestep
        # input_idx, body_param_idx, start, stride, part_size, end, axis
        TensorIteratorSliceInputDesc(2, 0, 0, 1, 1, -1, 0),
        # data
        TensorIteratorSliceInputDesc(3, 1, 0, 1, 1, -1, 0),
    ]
    ti_merged_input_desc = [
        # body prev/curr_cma
        TensorIteratorMergedInputDesc(4, 2, 0),
    ]
    ti_invariant_input_desc = [
        # body const one
        TensorIteratorInvariantInputDesc(5, 3),
    ]

    # TI outputs
    ti_body_output_desc = [
        # final average
        TensorIteratorBodyOutputDesc(0, 0, -1),
    ]
    ti_concat_output_desc = [
        # history of cma
        TensorIteratorConcatOutputDesc(1, 1, 0, 1, 1, -1, 0),
    ]

    node = ng.loop(
        trip_count,
        condition,
        ti_inputs,
        graph_body,
        ti_slice_input_desc,
        ti_merged_input_desc,
        ti_invariant_input_desc,
        ti_body_output_desc,
        ti_concat_output_desc,
        2,
        -1,
    )

    assert node.get_type_name() == "Loop"
    assert node.get_output_size() == 2
    # final average
    assert list(node.get_output_shape(0)) == [2, 2]
    # cma history
    assert list(node.get_output_shape(1)) == [16, 2, 2]


def test_roi_pooling():
    inputs = ng.parameter([2, 3, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.roi_pooling(inputs, coords, [6, 6], 0.0625, "Max")

    assert node.get_type_name() == "ROIPooling"
    assert node.get_output_size() == [6, 6]
    assert list(node.get_output_shape(0)) == [150, 3, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_psroi_pooling():
    inputs = ng.parameter([1, 72, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.psroi_pooling(inputs, coords, 2, 6, 0.0625, 0, 0, "average")

    assert node.get_type_name() == "PSROIPooling"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [150, 2, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_convert_like():
    parameter_data = ng.parameter([1, 2, 3, 4], name="data", dtype=np.float32)
    like = ng.constant(1, dtype=np.int8)

    node = ng.convert_like(parameter_data, like)

    assert node.get_type_name() == "ConvertLike"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2, 3, 4]
    assert node.get_output_element_type(0) == Type.i8


def test_bucketize():
    data = ng.parameter([4, 3, 2, 1], name="data", dtype=np.float32)
    buckets = ng.parameter([5], name="buckets", dtype=np.int64)

    node = ng.bucketize(data, buckets, "i32")

    assert node.get_type_name() == "Bucketize"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [4, 3, 2, 1]
    assert node.get_output_element_type(0) == Type.i32


def test_region_yolo():
    data = ng.parameter([1, 125, 13, 13], name="input", dtype=np.float32)
    num_coords = 4
    num_classes = 80
    num_regions = 1
    mask = [6, 7, 8]
    axis = 0
    end_axis = 3
    do_softmax = False

    node = ng.region_yolo(data, num_coords, num_classes, num_regions, do_softmax, mask, axis, end_axis)

    assert node.get_type_name() == "RegionYolo"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, (80 + 4 + 1) * 3, 13, 13]
    assert node.get_output_element_type(0) == Type.f32


def test_reorg_yolo():
    data = ng.parameter([2, 24, 34, 62], name="input", dtype=np.int32)
    stride = [2]

    node = ng.reorg_yolo(data, stride)

    assert node.get_type_name() == "ReorgYolo"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 96, 17, 31]
    assert node.get_output_element_type(0) == Type.i32


def test_embedding_bag_offsets_sum_1():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    offsets = ng.parameter([3], name="offsets", dtype=np.int64)
    default_index = ng.parameter([], name="default_index", dtype=np.int64)

    node = ng.embedding_bag_offsets_sum(emb_table, indices, offsets, default_index)

    assert node.get_type_name() == "EmbeddingBagOffsetsSum"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_segments_sum_all_inputs():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    segment_ids = ng.parameter([4], name="segment_ids", dtype=np.int64)
    num_segments = ng.parameter([], name="num_segments", dtype=np.int64)
    default_index = ng.parameter([], name="default_index", dtype=np.int64)
    per_sample_weights = ng.parameter([4], name="per_sample_weights", dtype=np.float32)

    node = ng.embedding_segments_sum(
        emb_table, indices, segment_ids, num_segments, default_index, per_sample_weights
    )

    assert node.get_type_name() == "EmbeddingSegmentsSum"
    assert node.get_output_size() == 1
    assert node.get_output_partial_shape(0).same_scheme(PartialShape([-1, 2]))
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_segments_sum_with_some_opt_inputs():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([4], name="indices", dtype=np.int64)
    segment_ids = ng.parameter([4], name="segment_ids", dtype=np.int64)
    num_segments = ng.parameter([], name="num_segments", dtype=np.int64)

    # only 1 out of 3 optional inputs
    node = ng.embedding_segments_sum(emb_table, indices, segment_ids, num_segments)

    assert node.get_type_name() == "EmbeddingSegmentsSum"
    assert node.get_output_size() == 1
    assert node.get_output_partial_shape(0).same_scheme(PartialShape([-1, 2]))
    assert node.get_output_element_type(0) == Type.f32


def test_embedding_bag_packed_sum():
    emb_table = ng.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = ng.parameter([3, 3], name="indices", dtype=np.int64)
    per_sample_weights = ng.parameter([3, 3], name="per_sample_weights", dtype=np.float32)

    # only 1 out of 3 optional inputs
    node = ng.embedding_bag_packed_sum(emb_table, indices, per_sample_weights)

    assert node.get_type_name() == "EmbeddingBagPackedSum"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32


@pytest.mark.parametrize("dtype", integral_np_types)
def test_interpolate_opset1(dtype):
    image_shape = [1, 3, 1024, 1024]
    output_shape = [64, 64]
    attributes = {
        "axes": [2, 3],
        "mode": "cubic",
        "pads_begin": np.array([2, 2], dtype=dtype),
    }

    image_node = ng.parameter(image_shape, dtype, name="Image")

    node = ng_opset1.interpolate(image_node, output_shape, attributes)
    expected_shape = [1, 3, 64, 64]

    assert node.get_type_name() == "Interpolate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_prior_box(int_dtype, fp_dtype):
    image_shape = np.array([64, 64], dtype=int_dtype)
    attributes = {
        "offset": fp_dtype(0),
        "min_size": np.array([2, 3], dtype=fp_dtype),
        "aspect_ratio": np.array([1.5, 2.0, 2.5], dtype=fp_dtype),
        "scale_all_sizes": False
    }

    layer_shape = ng.constant(np.array([32, 32], dtype=int_dtype), int_dtype)

    node = ng.prior_box(layer_shape, image_shape, attributes)

    assert node.get_type_name() == "PriorBox"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 20480]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.int8, np.float32),
        (np.int16, np.float32),
        (np.int32, np.float32),
        (np.int64, np.float32),
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.int32, np.float16),
        (np.int32, np.float64),
    ],
)
def test_prior_box_clustered(int_dtype, fp_dtype):
    image_size = np.array([64, 64], dtype=int_dtype)
    attributes = {
        "offset": fp_dtype(0.5),
        "width": np.array([4.0, 2.0, 3.2], dtype=fp_dtype),
        "height": np.array([1.0, 2.0, 1.0], dtype=fp_dtype),
    }

    output_size = ng.constant(np.array([19, 19], dtype=int_dtype), int_dtype)

    node = ng.prior_box_clustered(output_size, image_size, attributes)

    assert node.get_type_name() == "PriorBoxClustered"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 4332]


@pytest.mark.parametrize(
    "int_dtype, fp_dtype",
    [
        (np.uint8, np.float32),
        (np.uint16, np.float32),
        (np.uint32, np.float32),
        (np.uint64, np.float32),
        (np.uint32, np.float16),
        (np.uint32, np.float64),
    ],
)
def test_proposal(int_dtype, fp_dtype):
    attributes = {
        "base_size": int_dtype(1),
        "pre_nms_topn": int_dtype(20),
        "post_nms_topn": int_dtype(64),
        "nms_thresh": fp_dtype(0.34),
        "feat_stride": int_dtype(16),
        "min_size": int_dtype(32),
        "ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=fp_dtype),
        "scale": np.array([2, 3, 3, 4], dtype=fp_dtype),
    }
    batch_size = 7

    class_probs = ng.parameter([batch_size, 12, 34, 62], fp_dtype, "class_probs")
    bbox_deltas = ng.parameter([batch_size, 24, 34, 62], fp_dtype, "bbox_deltas")
    image_shape = ng.parameter([3], fp_dtype, "image_shape")
    node = ng.proposal(class_probs, bbox_deltas, image_shape, attributes)

    assert node.get_type_name() == "Proposal"
    assert node.get_output_size() == 2
    assert list(node.get_output_shape(0)) == [batch_size * attributes["post_nms_topn"], 5]


def test_tensor_iterator():
    from ngraph.utils.tensor_iterator_types import (
        GraphBody,
        TensorIteratorSliceInputDesc,
        TensorIteratorMergedInputDesc,
        TensorIteratorInvariantInputDesc,
        TensorIteratorBodyOutputDesc,
        TensorIteratorConcatOutputDesc,
    )

    #  Body parameters
    body_timestep = ng.parameter([], np.int32, "timestep")
    body_data_in = ng.parameter([1, 2, 2], np.float32, "body_in")
    body_prev_cma = ng.parameter([2, 2], np.float32, "body_prev_cma")
    body_const_one = ng.parameter([], np.int32, "body_const_one")

    # CMA = cumulative moving average
    prev_cum_sum = ng.multiply(ng.convert(body_timestep, "f32"), body_prev_cma)
    curr_cum_sum = ng.add(prev_cum_sum, ng.squeeze(body_data_in, [0]))
    elem_cnt = ng.add(body_const_one, body_timestep)
    curr_cma = ng.divide(curr_cum_sum, ng.convert(elem_cnt, "f32"))
    cma_hist = ng.unsqueeze(curr_cma, [0])

    # TI inputs
    data = ng.parameter([16, 2, 2], np.float32, "data")
    # Iterations count
    zero = ng.constant(0, dtype=np.int32)
    one = ng.constant(1, dtype=np.int32)
    initial_cma = ng.constant(np.zeros([2, 2], dtype=np.float32), dtype=np.float32)
    iter_cnt = ng.range(zero, np.int32(16), np.int32(1))
    ti_inputs = [iter_cnt, data, initial_cma, one]

    graph_body = GraphBody([body_timestep, body_data_in, body_prev_cma, body_const_one], [curr_cma, cma_hist])
    ti_slice_input_desc = [
        # timestep
        # input_idx, body_param_idx, start, stride, part_size, end, axis
        TensorIteratorSliceInputDesc(0, 0, 0, 1, 1, -1, 0),
        # data
        TensorIteratorSliceInputDesc(1, 1, 0, 1, 1, -1, 0),
    ]
    ti_merged_input_desc = [
        # body prev/curr_cma
        TensorIteratorMergedInputDesc(2, 2, 0),
    ]
    ti_invariant_input_desc = [
        # body const one
        TensorIteratorInvariantInputDesc(3, 3),
    ]

    # TI outputs
    ti_body_output_desc = [
        # final average
        TensorIteratorBodyOutputDesc(0, 0, -1),
    ]
    ti_concat_output_desc = [
        # history of cma
        TensorIteratorConcatOutputDesc(1, 1, 0, 1, 1, -1, 0),
    ]

    node = ng.tensor_iterator(
        ti_inputs,
        graph_body,
        ti_slice_input_desc,
        ti_merged_input_desc,
        ti_invariant_input_desc,
        ti_body_output_desc,
        ti_concat_output_desc,
    )

    assert node.get_type_name() == "TensorIterator"
    assert node.get_output_size() == 2
    # final average
    assert list(node.get_output_shape(0)) == [2, 2]
    # cma history
    assert list(node.get_output_shape(1)) == [16, 2, 2]


def test_read_value_opset5():
    init_value = ng_opset5.parameter([2, 2], name="init_value", dtype=np.int32)

    node = ng_opset5.read_value(init_value, "var_id_667")

    assert node.get_type_name() == "ReadValue"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.get_output_element_type(0) == Type.i32


def test_assign_opset5():
    input_data = ng_opset5.parameter([5, 7], name="input_data", dtype=np.int32)
    rv = ng_opset5.read_value(input_data, "var_id_667")
    node = ng_opset5.assign(rv, "var_id_667")

    assert node.get_type_name() == "Assign"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [5, 7]
    assert node.get_output_element_type(0) == Type.i32


def test_read_value():
    init_value = ng.parameter([2, 2], name="init_value", dtype=np.int32)

    node = ng.read_value(init_value, "var_id_667")

    assert node.get_type_name() == "ReadValue"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 2]
    assert node.get_output_element_type(0) == Type.i32


def test_assign():
    input_data = ng.parameter([5, 7], name="input_data", dtype=np.int32)
    rv = ng.read_value(input_data, "var_id_667")
    node = ng.assign(rv, "var_id_667")

    assert node.get_type_name() == "Assign"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [5, 7]
    assert node.get_output_element_type(0) == Type.i32


def test_extract_image_patches():
    image = ng.parameter([64, 3, 10, 10], name="image", dtype=np.int32)
    sizes = [3, 3]
    strides = [5, 5]
    rates = [1, 1]
    padding = "VALID"
    node = ng.extract_image_patches(image, sizes, strides, rates, padding)

    assert node.get_type_name() == "ExtractImagePatches"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [64, 27, 2, 2]
    assert node.get_output_element_type(0) == Type.i32


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_bidirectional(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128
    num_directions = 2
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "BIDIRECTIONAL"
    node = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_reverse(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "REVERSE"

    node_default = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "LSTMSequence"
    assert node_param.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lstm_sequence_operator_forward(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    C_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 4 * hidden_size, input_size]
    R_shape = [num_directions, 4 * hidden_size, hidden_size]
    B_shape = [num_directions, 4 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_C_t = ng.parameter(C_t_shape, name="C_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "forward"

    node_default = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "LSTMSequence"
    assert node_default.get_output_size() == 3

    activations = ["RELU", "tanh", "Sigmoid"]
    activation_alpha = [2.0]
    activation_beta = [1.0]
    clip = 0.5

    node = ng.lstm_sequence(
        parameter_X,
        parameter_H_t,
        parameter_C_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node.get_type_name() == "LSTMSequence"
    assert node.get_output_size() == 3


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gru_sequence_operator_bidirectional(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128
    num_directions = 2
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 3 * hidden_size, input_size]
    R_shape = [num_directions, 3 * hidden_size, hidden_size]
    B_shape = [num_directions, 3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "BIDIRECTIONAL"
    node = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node.get_type_name() == "GRUSequence"
    assert node.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22
    linear_before_reset = True
    B_shape = [num_directions, 4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    node_param = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
        linear_before_reset
    )

    assert node_param.get_type_name() == "GRUSequence"
    assert node_param.get_output_size() == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gru_sequence_operator_reverse(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 3 * hidden_size, input_size]
    R_shape = [num_directions, 3 * hidden_size, hidden_size]
    B_shape = [num_directions, 3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "REVERSE"

    node_default = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "GRUSequence"
    assert node_default.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22
    linear_before_reset = True
    B_shape = [num_directions, 4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    node_param = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
        linear_before_reset
    )

    assert node_param.get_type_name() == "GRUSequence"
    assert node_param.get_output_size() == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_gru_sequence_operator_forward(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, 3 * hidden_size, input_size]
    R_shape = [num_directions, 3 * hidden_size, hidden_size]
    B_shape = [num_directions, 3 * hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "forward"

    node_default = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "GRUSequence"
    assert node_default.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [2.0]
    activation_beta = [1.0]
    clip = 0.5
    linear_before_reset = True
    B_shape = [num_directions, 4 * hidden_size]
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    node = ng.gru_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
        linear_before_reset
    )

    assert node.get_type_name() == "GRUSequence"
    assert node.get_output_size() == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rnn_sequence_operator_bidirectional(dtype):
    batch_size = 1
    input_size = 16
    hidden_size = 128
    num_directions = 2
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, hidden_size, input_size]
    R_shape = [num_directions, hidden_size, hidden_size]
    B_shape = [num_directions, hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "BIDIRECTIONAL"
    node = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node.get_type_name() == "RNNSequence"
    assert node.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "RNNSequence"
    assert node_param.get_output_size() == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rnn_sequence_operator_reverse(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, hidden_size, input_size]
    R_shape = [num_directions, hidden_size, hidden_size]
    B_shape = [num_directions, hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "REVERSE"

    node_default = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "RNNSequence"
    assert node_default.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [1.0, 2.0, 3.0]
    activation_beta = [3.0, 2.0, 1.0]
    clip = 1.22

    node_param = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node_param.get_type_name() == "RNNSequence"
    assert node_param.get_output_size() == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rnn_sequence_operator_forward(dtype):
    batch_size = 2
    input_size = 4
    hidden_size = 3
    num_directions = 1
    seq_length = 2

    X_shape = [batch_size, seq_length, input_size]
    H_t_shape = [batch_size, num_directions, hidden_size]
    seq_len_shape = [batch_size]
    W_shape = [num_directions, hidden_size, input_size]
    R_shape = [num_directions, hidden_size, hidden_size]
    B_shape = [num_directions, hidden_size]

    parameter_X = ng.parameter(X_shape, name="X", dtype=dtype)
    parameter_H_t = ng.parameter(H_t_shape, name="H_t", dtype=dtype)
    parameter_seq_len = ng.parameter(seq_len_shape, name="seq_len", dtype=np.int32)
    parameter_W = ng.parameter(W_shape, name="W", dtype=dtype)
    parameter_R = ng.parameter(R_shape, name="R", dtype=dtype)
    parameter_B = ng.parameter(B_shape, name="B", dtype=dtype)

    direction = "forward"

    node_default = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
    )

    assert node_default.get_type_name() == "RNNSequence"
    assert node_default.get_output_size() == 2

    activations = ["RELU", "tanh"]
    activation_alpha = [2.0]
    activation_beta = [1.0]
    clip = 0.5

    node = ng.rnn_sequence(
        parameter_X,
        parameter_H_t,
        parameter_seq_len,
        parameter_W,
        parameter_R,
        parameter_B,
        hidden_size,
        direction,
        activations,
        activation_alpha,
        activation_beta,
        clip,
    )

    assert node.get_type_name() == "RNNSequence"
    assert node.get_output_size() == 2


def test_multiclass_nms():
    boxes_data = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1,
                           0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0,
                           0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0], dtype="float32")
    boxes_data = boxes_data.reshape([1, 6, 4])
    box = ng.constant(boxes_data, dtype=np.float)
    scores_data = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
                            0.95, 0.75, 0.6, 0.80, 0.5, 0.3], dtype="float32")
    scores_data = scores_data.reshape([1, 2, 6])
    score = ng.constant(scores_data, dtype=np.float)

    nms_node = ng.multiclass_nms(box, score, None, output_type="i32", nms_top_k=3,
                                 iou_threshold=0.5, score_threshold=0.0, sort_result_type="classid",
                                 nms_eta=1.0)

    assert nms_node.get_type_name() == "MulticlassNms"
    assert nms_node.get_output_size() == 3
    assert nms_node.outputs()[0].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(6)])
    assert nms_node.outputs()[1].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(1)])
    assert list(nms_node.outputs()[2].get_shape()) == [1, ]
    assert nms_node.get_output_element_type(0) == Type.f32
    assert nms_node.get_output_element_type(1) == Type.i32
    assert nms_node.get_output_element_type(2) == Type.i32

    boxes_data = np.array([[[7.55, 1.10, 18.28, 14.47],
                            [7.25, 0.47, 12.28, 17.77]],
                           [[4.06, 5.15, 16.11, 18.40],
                            [9.66, 3.36, 18.57, 13.26]],
                           [[6.50, 7.00, 13.33, 17.63],
                            [0.73, 5.34, 19.97, 19.97]]]).astype("float32")
    box = ng.constant(boxes_data, dtype=np.float)
    scores_data = np.array([[0.34, 0.66],
                            [0.45, 0.61],
                            [0.39, 0.59]]).astype("float32")
    score = ng.constant(scores_data, dtype=np.float)
    rois_num_data = np.array([3]).astype("int32")
    roisnum = ng.constant(rois_num_data, dtype=np.int)
    nms_node = ng.multiclass_nms(box, score, roisnum, output_type="i32", nms_top_k=3,
                                 iou_threshold=0.5, score_threshold=0.0, sort_result_type="classid",
                                 nms_eta=1.0)

    assert nms_node.get_type_name() == "MulticlassNms"
    assert nms_node.get_output_size() == 3
    assert nms_node.outputs()[0].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(6)])
    assert nms_node.outputs()[1].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(1)])
    assert list(nms_node.outputs()[2].get_shape()) == [1, ]
    assert nms_node.get_output_element_type(0) == Type.f32
    assert nms_node.get_output_element_type(1) == Type.i32
    assert nms_node.get_output_element_type(2) == Type.i32


def test_matrix_nms():
    boxes_data = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 1.0, 1.1,
                           0.0, -0.1, 1.0, 0.9, 0.0, 10.0, 1.0, 11.0,
                           0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0], dtype="float32")
    boxes_data = boxes_data.reshape([1, 6, 4])
    box = ng.constant(boxes_data, dtype=np.float)
    scores_data = np.array([0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
                            0.95, 0.75, 0.6, 0.80, 0.5, 0.3], dtype="float32")
    scores_data = scores_data.reshape([1, 2, 6])
    score = ng.constant(scores_data, dtype=np.float)

    nms_node = ng.matrix_nms(box, score, output_type="i32", nms_top_k=3,
                             score_threshold=0.0, sort_result_type="score", background_class=0,
                             decay_function="linear", gaussian_sigma=2.0, post_threshold=0.0)

    assert nms_node.get_type_name() == "MatrixNms"
    assert nms_node.get_output_size() == 3
    assert nms_node.outputs()[0].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(6)])
    assert nms_node.outputs()[1].get_partial_shape() == PartialShape([Dimension(0, 6), Dimension(1)])
    assert list(nms_node.outputs()[2].get_shape()) == [1, ]
    assert nms_node.get_output_element_type(0) == Type.f32
    assert nms_node.get_output_element_type(1) == Type.i32
    assert nms_node.get_output_element_type(2) == Type.i32


@pytest.mark.parametrize(
    ("boxes_shape", "scores_shape", "max_output_boxes", "expected_shape"),
    [
        ([1, 1000, 4], [1, 1, 1000], [1000], [PartialShape([Dimension(0, 1000), Dimension(3)]), PartialShape([Dimension(0, 1000), Dimension(3)])]),
        ([1, 700, 4], [1, 1, 700], [600], [PartialShape([Dimension(0, 600), Dimension(3)]), PartialShape([Dimension(0, 600), Dimension(3)])]),
        ([1, 300, 4], [1, 1, 300], [300], [PartialShape([Dimension(0, 300), Dimension(3)]), PartialShape([Dimension(0, 300), Dimension(3)])]),
    ],
)
def test_non_max_suppression(boxes_shape, scores_shape, max_output_boxes, expected_shape):
    boxes_parameter = ng.parameter(boxes_shape, name="Boxes", dtype=np.float32)
    scores_parameter = ng.parameter(scores_shape, name="Scores", dtype=np.float32)

    node = ng.non_max_suppression(boxes_parameter, scores_parameter, make_constant_node(max_output_boxes, np.int64))
    assert node.get_type_name() == "NonMaxSuppression"
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0) == expected_shape[0]
    assert node.get_output_partial_shape(1) == expected_shape[1]
    assert list(node.get_output_shape(2)) == [1]


@pytest.mark.parametrize(
    ("boxes_shape", "scores_shape", "max_output_boxes", "iou_threshold", "score_threshold", "soft_nms_sigma", "expected_shape"),
    [
        ([1, 100, 4], [1, 1, 100], [100], 0.1, 0.4, 0.5, [PartialShape([Dimension(0, 100), Dimension(3)]), PartialShape([Dimension(0, 100), Dimension(3)])]),
        ([1, 700, 4], [1, 1, 700], [600], 0.1, 0.4, 0.5, [PartialShape([Dimension(0, 600), Dimension(3)]), PartialShape([Dimension(0, 600), Dimension(3)])]),
        ([1, 300, 4], [1, 1, 300], [300], 0.1, 0.4, 0.5, [PartialShape([Dimension(0, 300), Dimension(3)]), PartialShape([Dimension(0, 300), Dimension(3)])]),
    ],
)
def test_non_max_suppression_non_default_args(boxes_shape, scores_shape, max_output_boxes, iou_threshold, score_threshold, soft_nms_sigma, expected_shape):
    boxes_parameter = ng.parameter(boxes_shape, name="Boxes", dtype=np.float32)
    scores_parameter = ng.parameter(scores_shape, name="Scores", dtype=np.float32)

    max_output_boxes = make_constant_node(max_output_boxes, np.int64)
    iou_threshold = make_constant_node(iou_threshold, np.float32)
    score_threshold = make_constant_node(score_threshold, np.float32)
    soft_nms_sigma = make_constant_node(soft_nms_sigma, np.float32)

    node = ng.non_max_suppression(boxes_parameter, scores_parameter, max_output_boxes, iou_threshold, score_threshold, soft_nms_sigma)
    assert node.get_type_name() == "NonMaxSuppression"
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0) == expected_shape[0]
    assert node.get_output_partial_shape(1) == expected_shape[1]
    assert list(node.get_output_shape(2)) == [1]


def test_slice():
    data_shape = [10, 7, 2, 13]
    data = ng.parameter(data_shape, name="input", dtype=np.float32)

    start = ng.constant(np.array([2, 0, 0], dtype=np.int32))
    stop = ng.constant(np.array([9, 7, 2], dtype=np.int32))
    step = ng.constant(np.array([2, 1, 1], dtype=np.int32))

    node_default_axes = ng.slice(data, start, stop, step)

    assert node_default_axes.get_type_name() == "Slice"
    assert node_default_axes.get_output_size() == 1
    assert node_default_axes.get_output_element_type(0) == Type.f32
    assert tuple(node_default_axes.get_output_shape(0)) == np.zeros(data_shape)[2:9:2, ::, 0:2:1].shape

    start = ng.constant(np.array([0, 2], dtype=np.int32))
    stop = ng.constant(np.array([2, 9], dtype=np.int32))
    step = ng.constant(np.array([1, 2], dtype=np.int32))
    axes = ng.constant(np.array([-2, 0], dtype=np.int32))

    node = ng.slice(data, start, stop, step, axes)

    assert node.get_type_name() == "Slice"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert tuple(node.get_output_shape(0)) == np.zeros(data_shape)[2:9:2, ::, 0:2:1].shape


def test_i420_to_bgr():
    expected_output_shape = [1, 480, 640, 3]

    # # Single plane (one arg)
    arg_single_plane = ng.parameter([1, 720, 640, 1], name="input", dtype=np.float32)
    node_single_plane = ng.i420_to_bgr(arg_single_plane)

    assert node_single_plane.get_type_name() == "I420toBGR"
    assert node_single_plane.get_output_size() == 1
    assert node_single_plane.get_output_element_type(0) == Type.f32
    assert list(node_single_plane.get_output_shape(0)) == expected_output_shape

    # Separate planes (three args)
    arg_y = ng.parameter([1, 480, 640, 1], name="input_y", dtype=np.float32)
    arg_u = ng.parameter([1, 240, 320, 1], name="input_u", dtype=np.float32)
    arg_v = ng.parameter([1, 240, 320, 1], name="input_v", dtype=np.float32)

    node_separate_planes = ng.i420_to_bgr(arg_y, arg_u, arg_v)

    assert node_separate_planes.get_type_name() == "I420toBGR"
    assert node_separate_planes.get_output_size() == 1
    assert node_separate_planes.get_output_element_type(0) == Type.f32
    assert list(node_separate_planes.get_output_shape(0)) == expected_output_shape

    # Incorrect inputs number
    with pytest.raises(UserInputError, match=r".*Operation I420toBGR*."):
        node_separate_planes = ng.i420_to_bgr(arg_y, arg_v)

    with pytest.raises(UserInputError, match=r".*Operation I420toBGR*."):
        node_separate_planes = ng.i420_to_bgr(arg_single_plane, None, arg_v)


def test_i420_to_rgb():
    expected_output_shape = [1, 480, 640, 3]

    # # Single plane (one arg)
    arg_single_plane = ng.parameter([1, 720, 640, 1], name="input", dtype=np.float32)
    node_single_plane = ng.i420_to_rgb(arg_single_plane)

    assert node_single_plane.get_type_name() == "I420toRGB"
    assert node_single_plane.get_output_size() == 1
    assert node_single_plane.get_output_element_type(0) == Type.f32
    assert list(node_single_plane.get_output_shape(0)) == expected_output_shape

    # Separate planes (three args)
    arg_y = ng.parameter([1, 480, 640, 1], name="input_y", dtype=np.float32)
    arg_u = ng.parameter([1, 240, 320, 1], name="input_u", dtype=np.float32)
    arg_v = ng.parameter([1, 240, 320, 1], name="input_v", dtype=np.float32)

    node_separate_planes = ng.i420_to_rgb(arg_y, arg_u, arg_v)

    assert node_separate_planes.get_type_name() == "I420toRGB"
    assert node_separate_planes.get_output_size() == 1
    assert node_separate_planes.get_output_element_type(0) == Type.f32
    assert list(node_separate_planes.get_output_shape(0)) == expected_output_shape

    with pytest.raises(UserInputError, match=r".*Operation I420toRGB*."):
        node_separate_planes = ng.i420_to_rgb(arg_y, arg_v)

    with pytest.raises(UserInputError, match=r".*Operation I420toRGB*."):
        node_separate_planes = ng.i420_to_rgb(arg_single_plane, None, arg_v)


def test_nv12_to_bgr():
    expected_output_shape = [1, 480, 640, 3]

    # # Single plane (one arg)
    arg_single_plane = ng.parameter([1, 720, 640, 1], name="input", dtype=np.float32)
    node_single_plane = ng.nv12_to_bgr(arg_single_plane)

    assert node_single_plane.get_type_name() == "NV12toBGR"
    assert node_single_plane.get_output_size() == 1
    assert node_single_plane.get_output_element_type(0) == Type.f32
    assert list(node_single_plane.get_output_shape(0)) == expected_output_shape

    # Separate planes (two args)
    arg_y = ng.parameter([1, 480, 640, 1], name="input_y", dtype=np.float32)
    arg_uv = ng.parameter([1, 240, 320, 2], name="input_uv", dtype=np.float32)

    node_separate_planes = ng.nv12_to_bgr(arg_y, arg_uv)

    assert node_separate_planes.get_type_name() == "NV12toBGR"
    assert node_separate_planes.get_output_size() == 1
    assert node_separate_planes.get_output_element_type(0) == Type.f32
    assert list(node_separate_planes.get_output_shape(0)) == expected_output_shape


def test_nv12_to_rgb():
    expected_output_shape = [1, 480, 640, 3]

    # # Single plane (one arg)
    arg_single_plane = ng.parameter([1, 720, 640, 1], name="input", dtype=np.float32)
    node_single_plane = ng.nv12_to_rgb(arg_single_plane)

    assert node_single_plane.get_type_name() == "NV12toRGB"
    assert node_single_plane.get_output_size() == 1
    assert node_single_plane.get_output_element_type(0) == Type.f32
    assert list(node_single_plane.get_output_shape(0)) == expected_output_shape

    # Separate planes (two args)
    arg_y = ng.parameter([1, 480, 640, 1], name="input_y", dtype=np.float32)
    arg_uv = ng.parameter([1, 240, 320, 2], name="input_uv", dtype=np.float32)

    node_separate_planes = ng.nv12_to_rgb(arg_y, arg_uv)

    assert node_separate_planes.get_type_name() == "NV12toRGB"
    assert node_separate_planes.get_output_size() == 1
    assert node_separate_planes.get_output_element_type(0) == Type.f32
    assert list(node_separate_planes.get_output_shape(0)) == expected_output_shape


def test_softsign():
    input_shape = [2, 4, 8, 16]

    param = ng.parameter(input_shape, name="input")
    node = ng.softsign(param, input_shape)

    assert node.get_type_name() == "SoftSign"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == input_shape
    assert node.get_output_element_type(0) == Type.f32


def test_rdft():
    param = ng.parameter([5, 3, 4], name="input")
    axes = ng.constant(np.array([0, 1]))
    signal_size = ng.constant(np.array([1, 2]))
    node = ng.rdft(param, axes, signal_size)

    assert node.get_type_name() == "RDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2, 4, 2]
    assert node.get_output_element_type(0) == Type.f32


def test_irdft():
    param = ng.parameter([5, 3, 4, 2], name="input")
    axes = ng.constant(np.array([0, 1]))
    signal_size = ng.constant(np.array([1, 2]))
    node = ng.irdft(param, axes, signal_size)

    assert node.get_type_name() == "IRDFT"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2, 4]
    assert node.get_output_element_type(0) == Type.f32


def test_generate_proposals():
    im_info_shape = [1, 3]
    anchors_shape = [4, 4, 3, 4]
    deltas_shape = [1, 12, 4, 4]
    scores_shape = [1, 3, 4, 4]

    im_info_param = ng.parameter(im_info_shape, name="im_info")
    anchors_param = ng.parameter(anchors_shape, name="anchors")
    deltas_param = ng.parameter(deltas_shape, name="deltas")
    scores_param = ng.parameter(scores_shape, name="scores")

    node = ng.generate_proposals(im_info_param,
                                 anchors_param,
                                 deltas_param,
                                 scores_param,
                                 min_size=1.0,
                                 nms_threshold=0.5,
                                 pre_nms_count=200,
                                 post_nms_count=100,
                                 normalized=False,
                                 nms_eta=1.0,
                                 roi_num_type="i32")

    assert node.get_type_name() == "GenerateProposals"
    assert node.get_output_size() == 3
    assert node.get_output_partial_shape(0).same_scheme(PartialShape([-1, 4]))
    assert node.get_output_partial_shape(1).same_scheme(PartialShape([-1]))
    assert node.get_output_partial_shape(2).same_scheme(PartialShape([1]))
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.f32
    assert node.get_output_element_type(2) == Type.i32


def test_grid_sample_default():
    img = ng.parameter([1, 3, 100, 100], dtype=np.int32, name="image")
    grid = ng.parameter([1, 10, 10, 2], dtype=np.float32, name="grid")

    node = ng.grid_sample(img, grid, {})

    assert node.get_type_name() == "GridSample"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 3, 10, 10]
    assert node.get_output_element_type(0) == Type.i32


def test_grid_sample_custom_attributes():
    img = ng.parameter([1, 3, 100, 100], dtype=np.int32, name="image")
    grid = ng.parameter([1, 5, 6, 2], dtype=np.float32, name="grid")

    attributes = {
        "align_corners": True,
        "mode": "nearest",
        "padding_mode": "reflection"
    }

    node = ng.grid_sample(img, grid, attributes)

    assert node.get_type_name() == "GridSample"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 3, 5, 6]
    assert node.get_output_element_type(0) == Type.i32

    node_attributes = node.get_attributes()
    assert node_attributes["align_corners"] is True
    assert node_attributes["mode"] == "nearest"
    assert node_attributes["padding_mode"] == "reflection"


@pytest.mark.parametrize(
    ("expected_shape", "shape_calculation_mode"),
    [
        ([1, 3, 64, 64], "scales"),
        ([1, 3, 256, 256], "sizes"),
    ],
)
@pytest.mark.parametrize("dtype", np_types)
def test_interpolate_opset10(dtype, expected_shape, shape_calculation_mode):

    image_shape = [1, 3, 1024, 1024]
    image_node = ng.parameter(image_shape, dtype, name="Image")
    output_shape = [256, 256]
    scales = np.array([1 / 16, 1 / 16], dtype=np.float32)
    axes = [2, 3]
    mode = "cubic"

    node = ng_opset10.interpolate(image=image_node, output_shape=output_shape, scales=scales,
                                  axes=axes,
                                  mode=mode, shape_calculation_mode=shape_calculation_mode)
    assert node.get_type_name() == "Interpolate"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape

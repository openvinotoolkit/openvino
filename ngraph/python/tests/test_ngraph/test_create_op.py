# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from _pyngraph import PartialShape

import ngraph as ng
import ngraph.opset1 as ng_opset1
import ngraph.opset5 as ng_opset5
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
                         ],)
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
def test_interpolate(dtype):
    image_shape = [1, 3, 1024, 1024]
    output_shape = [64, 64]
    attributes = {
        "axes": [2, 3],
        "mode": "cubic",
        "pads_begin": np.array([2, 2], dtype=dtype),
    }

    image_node = ng.parameter(image_shape, dtype, name="Image")

    node = ng.interpolate(image_node, output_shape, attributes)
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
def test_detection_output(int_dtype, fp_dtype):
    attributes = {
        "num_classes": int_dtype(85),
        "keep_top_k": np.array([64], dtype=int_dtype),
        "nms_threshold": fp_dtype(0.645),
    }

    box_logits = ng.parameter([4, 8], fp_dtype, "box_logits")
    class_preds = ng.parameter([4, 170], fp_dtype, "class_preds")
    proposals = ng.parameter([4, 2, 10], fp_dtype, "proposals")
    aux_class_preds = ng.parameter([4, 4], fp_dtype, "aux_class_preds")
    aux_box_preds = ng.parameter([4, 8], fp_dtype, "aux_box_preds")

    node = ng.detection_output(box_logits, class_preds, proposals, attributes, aux_class_preds, aux_box_preds)

    assert node.get_type_name() == "DetectionOutput"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 1, 256, 7]


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

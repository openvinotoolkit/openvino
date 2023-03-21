# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import numpy as np
from openvino.runtime import serialize
from openvino._offline_transformations import (
    apply_moc_transformations,
    apply_pot_transformations,
    apply_low_latency_transformation,
    apply_pruning_transformation,
    apply_make_stateful_transformation,
    compress_model_transformation,
    convert_sequence_to_tensor_iterator_transformation,
    apply_fused_names_cleanup,
)

from openvino.runtime import Model, PartialShape, Core
import openvino.runtime as ov

from tests.test_utils.test_utils import create_filename_for_test


def get_relu_model():
    param = ov.opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    param.get_output_tensor(0).set_names({"parameter"})
    relu = ov.opset8.relu(param)
    res = ov.opset8.result(relu, name="result")
    res.get_output_tensor(0).set_names({"result"})
    return Model([res], [param], "test")


def get_lstm_sequence_model():
    parameter_x = ov.opset9.parameter([1, 2, 16], name="X")
    parameter_y = ov.opset9.parameter([1, 1, 128], name="Y")
    parameter_z = ov.opset9.parameter([1, 1, 128], name="Z")
    seq_lengths = ov.opset9.constant(np.array([2]), dtype=np.int32)

    w_val = np.zeros(shape=[1, 512, 16])
    r_val = np.zeros(shape=[1, 512, 128])
    b_val = np.zeros(shape=[1, 512])

    const_w = ov.opset9.constant(value=w_val, dtype=np.float32)
    const_r = ov.opset9.constant(value=r_val, dtype=np.float32)
    const_b = ov.opset9.constant(value=b_val, dtype=np.float32)

    lstm_sequence = ov.opset9.lstm_sequence(parameter_x, parameter_y, parameter_z, seq_lengths, const_w, const_r, const_b, 128, "FORWARD")
    y_out = ov.opset9.result(lstm_sequence.output(0))
    ho = ov.opset9.result(lstm_sequence.output(1))
    co = ov.opset9.result(lstm_sequence.output(2))

    model = Model([y_out, ho, co], [parameter_x, parameter_y, parameter_z])
    return model


def get_rnn_sequence_model():
    parameter_x = ov.opset9.parameter([1, 2, 16], name="X")
    parameter_y = ov.opset9.parameter([1, 1, 128], name="Y")
    seq_lengths = ov.opset9.constant(np.array([2]), dtype=np.int32)

    w_val = np.zeros(shape=[1, 128, 16])
    r_val = np.zeros(shape=[1, 128, 128])
    b_val = np.zeros(shape=[1, 128])

    const_w = ov.opset9.constant(value=w_val, dtype=np.float32)
    const_r = ov.opset9.constant(value=r_val, dtype=np.float32)
    const_b = ov.opset9.constant(value=b_val, dtype=np.float32)

    rnn_sequence = ov.opset9.rnn_sequence(parameter_x, parameter_y, seq_lengths, const_w, const_r, const_b, 128, "FORWARD")
    y_out = ov.opset9.result(rnn_sequence.output(0))
    model = Model([y_out], [parameter_x, parameter_y])

    return model


def get_gru_sequence_model():
    parameter_x = ov.opset9.parameter([1, 2, 16], name="X")
    parameter_y = ov.opset9.parameter([1, 1, 128], name="Y")
    seq_lengths = ov.opset9.constant(np.array([2]), dtype=np.int32)

    w_val = np.zeros(shape=[1, 384, 16])
    r_val = np.zeros(shape=[1, 384, 128])
    b_val = np.zeros(shape=[1, 384])

    const_w = ov.opset9.constant(value=w_val, dtype=np.float32)
    const_r = ov.opset9.constant(value=r_val, dtype=np.float32)
    const_b = ov.opset9.constant(value=b_val, dtype=np.float32)

    gru_sequence = ov.opset9.gru_sequence(parameter_x, parameter_y, seq_lengths, const_w, const_r, const_b, 128, "FORWARD")
    y_out = ov.opset9.result(gru_sequence.output(0))
    ho = ov.opset9.result(gru_sequence.output(1))
    model = Model([y_out, ho], [parameter_x, parameter_y])

    return model


def test_moc_transformations():
    model = get_relu_model()

    apply_moc_transformations(model, False)

    assert model is not None
    assert len(model.get_ops()) == 3


def test_moc_with_smart_reshape():
    model = get_relu_model()

    apply_moc_transformations(model, cf=False, smart_reshape=True)

    assert model is not None
    assert len(model.get_ops()) == 3


def test_pot_transformations():
    model = get_relu_model()

    apply_pot_transformations(model, "GNA")

    assert model is not None
    assert len(model.get_ops()) == 3


def test_low_latency_transformation():
    model = get_relu_model()

    apply_low_latency_transformation(model, True)

    assert model is not None
    assert len(model.get_ops()) == 3


def test_pruning_transformation():
    model = get_relu_model()

    apply_pruning_transformation(model)

    assert model is not None
    assert len(model.get_ops()) == 3


def test_make_stateful_transformations():
    model = get_relu_model()

    apply_make_stateful_transformation(model, {"parameter": "result"})

    assert model is not None
    assert len(model.get_parameters()) == 0
    assert len(model.get_results()) == 0


def test_fused_names_cleanup():
    model = get_relu_model()

    for node in model.get_ops():
        node.get_rt_info()["fused_names_0"] = "test_op_name"

    apply_fused_names_cleanup(model)

    assert model is not None
    assert len(model.get_ops()) == 3

    for node in model.get_ops():
        assert len(node.get_rt_info()) == 0


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.mark.parametrize("is_path_xml, is_path_bin", [  # noqa: PT006
    (True, True),
    (True, False),
    (False, True),
    (False, False),
],
)
def test_serialize_pass_v2(request, tmp_path, is_path_xml, is_path_bin):
    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name,
                                                  tmp_path,
                                                  is_path_xml,
                                                  is_path_bin)
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path)

    assert func is not None

    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_model.get_parameters()
    assert func.get_ordered_ops() == res_model.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_compress_model_transformation():
    node_constant = ov.opset8.constant(np.array([[0.0, 0.1, -0.1], [-2.5, 2.5, 3.0]], dtype=np.float32))
    node_ceil = ov.opset8.ceiling(node_constant)
    model = Model(node_ceil, [], "TestModel")
    elem_type = model.get_ordered_ops()[0].get_element_type().get_type_name()
    assert elem_type == "f32"
    compress_model_transformation(model)

    assert model is not None
    elem_type = model.get_ordered_ops()[0].get_element_type().get_type_name()
    assert elem_type == "f16"


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.mark.parametrize("is_path_xml, is_path_bin", [  # noqa: PT006
    (True, True),
    (True, False),
    (False, True),
    (False, False),
],
)
def test_version_default(request, tmp_path, is_path_xml, is_path_bin):
    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name,
                                                  tmp_path,
                                                  is_path_xml,
                                                  is_path_bin)
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path)
    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_model.get_parameters()
    assert func.get_ordered_ops() == res_model.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
@pytest.mark.parametrize("is_path_xml, is_path_bin", [  # noqa: PT006
    (True, True),
    (True, False),
    (False, True),
    (False, False),
],
)
def test_serialize_default_bin(request, tmp_path, is_path_xml, is_path_bin):
    xml_path, bin_path = create_filename_for_test(request.node.name,
                                                  tmp_path,
                                                  is_path_xml,
                                                  is_path_bin)
    model = get_relu_model()
    serialize(model, xml_path)
    assert os.path.exists(bin_path)
    os.remove(xml_path)
    os.remove(bin_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_version_ir_v10(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name, tmp_path)
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path, "IR_V10")
    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_model.get_parameters()
    assert func.get_ordered_ops() == res_model.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


# request - https://docs.pytest.org/en/7.1.x/reference/reference.html#request
def test_version_ir_v11(request, tmp_path):
    core = Core()
    xml_path, bin_path = create_filename_for_test(request.node.name, tmp_path)
    shape = [100, 100, 2]
    parameter_a = ov.opset8.parameter(shape, dtype=np.float32, name="A")
    parameter_b = ov.opset8.parameter(shape, dtype=np.float32, name="B")
    model = ov.opset8.floor(ov.opset8.minimum(ov.opset8.abs(parameter_a), parameter_b))
    func = Model(model, [parameter_a, parameter_b], "Model")

    serialize(func, xml_path, bin_path, "IR_V11")
    res_model = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_model.get_parameters()
    assert func.get_ordered_ops() == res_model.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)


def test_convert_lstm_to_tensor_iterator():
    model = get_lstm_sequence_model()
    ops_types = [op.get_type_name() for op in model.get_ops()]
    # assert that LSTM sequence is present in the model
    assert "TensorIterator" not in ops_types
    assert "LSTMSequence" in ops_types
    # assert that LSTM sequence got transformed into TensorIterator
    convert_sequence_to_tensor_iterator_transformation(model)
    ops_types = [op.get_type_name() for op in model.get_ops()]
    assert "LSTMSequence" not in ops_types
    assert "TensorIterator" in ops_types


def test_convert_rnn_to_tensor_iterator():
    model = get_rnn_sequence_model()
    ops_types = [op.get_type_name() for op in model.get_ops()]
    # assert that RNN sequence is present in the model
    assert "TensorIterator" not in ops_types
    assert "RNNSequence" in ops_types
    convert_sequence_to_tensor_iterator_transformation(model)
    ops_types = [op.get_type_name() for op in model.get_ops()]
    # assert that RNN sequence got transformed into TensorIterator
    assert "RNNSequence" not in ops_types
    assert "TensorIterator" in ops_types


def test_convert_gru_to_tensor_iterator():
    model = get_gru_sequence_model()
    ops_types = [op.get_type_name() for op in model.get_ops()]
    # assert that GRU sequence is present in the model
    assert "TensorIterator" not in ops_types
    assert "GRUSequence" in ops_types
    convert_sequence_to_tensor_iterator_transformation(model)
    ops_types = [op.get_type_name() for op in model.get_ops()]
    # assert that GRU sequence got transformed into TensorIterator
    assert "GRUSequence" not in ops_types
    assert "TensorIterator" in ops_types


def test_flush_fp32_subnormals_to_zero():
    parameter = ov.opset10.parameter([1, 8], name="X")
    subnorm_val = -2.0e-45

    weights = ov.opset10.constant(np.array([0.0, 1.0, 2.0, 3.0, subnorm_val, subnorm_val, subnorm_val, subnorm_val]),
                                  dtype=np.float32)
    add_node = ov.opset10.add(parameter, weights)

    result = ov.opset10.result(add_node)
    model = Model([result], [parameter])

    apply_moc_transformations(model, cf=False, smart_reshape=True)  # apply_flush_fp32_subnormals_to_zero is called inside

    assert np.all(weights.data[4:8] != subnorm_val)
    assert np.all(weights.data[4:8] == 0.0)

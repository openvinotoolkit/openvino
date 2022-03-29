# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import numpy as np

from openvino.runtime import Model, PartialShape, Shape, opset8, Core
from openvino.runtime.passes import Manager, ConstantFolding, MakeStateful,\
    ConvertFP32ToFP16, LowLatency2, Serialize

from utils.utils import count_ops, get_test_function


def get_model():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    param.get_output_tensor(0).set_names({"parameter"})
    relu = opset8.relu(param)
    reshape = opset8.reshape(relu, opset8.shape_of(relu), False)
    res = opset8.result(reshape, name="result")
    res.get_output_tensor(0).set_names({"result"})
    return Model([res], [param], "test")


def test_make_stateful():
    model = get_model()

    m = Manager()
    p = MakeStateful({"parameter": "result"})
    m.register_pass(p)
    m.run_passes(model)

    assert model is not None
    assert len(model.get_parameters()) == 0
    assert len(model.get_results()) == 0


def test_constant_folding():
    model = get_model()

    m = Manager()
    m.register_pass(ConstantFolding())
    m.run_passes(model)

    assert model is not None
    assert count_ops(model, "ShapeOf") == [0]


def test_convert_precision():
    model = get_model()

    m = Manager()
    m.register_pass(ConvertFP32ToFP16())
    m.run_passes(model)

    assert model is not None
    # TODO: fix bug 82773 with float16 type comparison
    # assert model.get_parameters()[0].get_element_type() == np.float16


def test_low_latency2():
    X = opset8.parameter(Shape([32, 40, 10]), np.float32, "X")
    Y = opset8.parameter(Shape([32, 40, 10]), np.float32, "Y")
    M = opset8.parameter(Shape([32, 2, 10]), np.float32, "M")

    X_i = opset8.parameter(Shape([32, 2, 10]), np.float32, "X_i")
    Y_i = opset8.parameter(Shape([32, 2, 10]), np.float32, "Y_i")
    M_body = opset8.parameter(Shape([32, 2, 10]), np.float32, "M_body")

    sum = opset8.add(X_i, Y_i)
    Zo = opset8.multiply(sum, M_body)

    body = Model([Zo], [X_i, Y_i, M_body], "body_function")

    ti = opset8.tensor_iterator()
    ti.set_body(body)
    ti.set_sliced_input(X_i, X.output(0), 0, 2, 2, 39, 1)
    ti.set_sliced_input(Y_i, Y.output(0), 0, 2, 2, -1, 1)
    ti.set_invariant_input(M_body, M.output(0))

    out0 = ti.get_iter_value(Zo.output(0), -1)
    out1 = ti.get_concatenated_slices(Zo.output(0), 0, 2, 2, 39, 1)

    result0 = opset8.result(out0)
    result1 = opset8.result(out1)

    model = Model([result0, result1], [X, Y, M])

    m = Manager()
    m.register_pass(LowLatency2())
    m.run_passes(model)

    # TODO: create TI which will be transformed by LowLatency2
    assert count_ops(model, "TensorIterator") == [1]


def test_serialize_pass():
    core = Core()
    xml_path = "serialized_function.xml"
    bin_path = "serialized_function.bin"

    func = get_test_function()

    m = Manager()
    m.register_pass(Serialize(xml_path, bin_path))
    m.run_passes(func)

    assert func is not None

    res_func = core.read_model(model=xml_path, weights=bin_path)

    assert func.get_parameters() == res_func.get_parameters()
    assert func.get_ordered_ops() == res_func.get_ordered_ops()

    os.remove(xml_path)
    os.remove(bin_path)

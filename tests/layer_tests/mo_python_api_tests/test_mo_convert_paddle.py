# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.mo_convert_test_class import CommonMOConvertTest

import openvino.runtime as ov
from openvino.runtime import PartialShape, Model

def make_pd_dynamic_graph_model():
    import paddle
    paddle.disable_static()
    class NeuralNetwork(paddle.nn.Layer):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.relu_sigmoid_stack = paddle.nn.Sequential(
                paddle.nn.ReLU(),
                paddle.nn.Sigmoid())
        def forward(self, input):
            return self.relu_sigmoid_stack(input)
    return NeuralNetwork()

def make_pd_static_graph_model(shape):
    import paddle
    import paddle.nn

    paddle.enable_static()

    x = paddle.static.data(name="x", shape=shape)
    y = paddle.static.data(name="y", shape=shape)
    relu = paddle.nn.ReLU()
    sigmoid = paddle.nn.Sigmoid()
    y = sigmoid(relu(x))
    
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())
    return exe, x, y

def make_pd_hapi_graph_model(shape):
    import paddle
    paddle.disable_static()
    from paddle.static import InputSpec
    net = paddle.nn.Sequential(
        paddle.nn.ReLU(),
        paddle.nn.Sigmoid())
    input = InputSpec(shape, 'float32', 'x')
    label = InputSpec(shape, 'float32', 'label')
    
    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(learning_rate=1e-3,
        parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())
    return model

def make_ref_graph_model(shape, dtype=np.float32):
    shape = PartialShape(shape)
    param = ov.opset8.parameter(shape, name="x", dtype=dtype)

    relu = ov.opset8.relu(param)
    sigm = ov.opset8.sigmoid(relu)

    model = Model([sigm], [param], "test")
    return model

def create_paddle_dynamic_module(tmp_dir):
    import paddle
    shape = [2,3,4]
    pd_model = make_pd_dynamic_graph_model()
    ref_model = make_ref_graph_model(shape)

    x = paddle.static.InputSpec(shape=shape, dtype='float32', name='x')
    return pd_model, ref_model, {"example_input": [x]}

def create_paddle_static_module(tmp_dir):
    shape = [2,3,4]
    pd_model, x, y = make_pd_static_graph_model(shape)
    ref_model = make_ref_graph_model(shape)

    return pd_model, ref_model, {"example_input": [x], "example_output": [y]}

def create_paddle_hapi_module(tmp_dir):
    shape = [2,3,4]
    pd_model = make_pd_hapi_graph_model(shape)
    ref_model = make_ref_graph_model(shape)

    return pd_model, ref_model, {}

class TestMoConvertPaddle(CommonMOConvertTest):
    test_data = [
        'create_paddle_dynamic_module',
        'create_paddle_static_module',
        'create_paddle_hapi_module'
    ]
    @pytest.mark.skip(reason="Paddlepaddle has incompatible protobuf. Ticket: 95904")
    @pytest.mark.parametrize("create_model", test_data)
    def test_mo_import_from_memory_paddle_fe(self, create_model, ie_device, precision, ir_version,
                                             temp_dir):
        fw_model, graph_ref, mo_params = eval(create_model)(temp_dir)
        test_params = {'input_model': fw_model}
        if mo_params is not None:
            test_params.update(mo_params)
        test_params.update({'use_convert_model_from_mo': True})
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)

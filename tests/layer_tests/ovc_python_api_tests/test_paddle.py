# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest

import numpy as np
import openvino.runtime as ov
import pytest
from openvino.runtime import PartialShape, Model
from openvino.test_utils import compare_functions

from common.mo_convert_test_class import CommonMOConvertTest


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

    return pd_model, ref_model, {"example_input": [x], "output": [y]}

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
        self._test_by_ref_graph(temp_dir, test_params, graph_ref, compare_tensor_names=False)


class TestPaddleConversionParams(CommonMOConvertTest):
    paddle_is_imported = False
    try:
        import paddle
        paddle_is_imported = True
    except ImportError:
        pass

    test_data = [
        {'params_test': {'input': paddle.shape(paddle.to_tensor(np.random.rand(2, 3, 4)))},
         'fw_model': make_pd_hapi_graph_model([1, 2]),
         'ref_model': make_ref_graph_model([2, 3, 4])},
        {'params_test': {'input': paddle.to_tensor(np.random.rand(5, 6)).shape},
         'fw_model': make_pd_hapi_graph_model([1, 2, 3]),
         'ref_model': make_ref_graph_model([5, 6])},
        {'params_test': {'input': (paddle.to_tensor(np.random.rand(4, 2, 7)).shape, paddle.int32)},
         'fw_model': make_pd_hapi_graph_model([2, 3]),
         'ref_model': make_ref_graph_model([4, 2, 7], np.int32)},
    ] if paddle_is_imported else []

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_conversion_params(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_legacy_frontend):
        fw_model = params['fw_model']
        test_params = params['params_test']
        ref_model = params['ref_model']

        test_params.update({'input_model': fw_model})
        self._test_by_ref_graph(temp_dir, test_params, ref_model, compare_tensor_names=False)


class TestUnicodePathsPaddle(unittest.TestCase):
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unicode_paths(self):
        import os
        try:
            import paddle
        except ImportError:
            return

        paddle.enable_static()
        test_directory = os.path.dirname(os.path.realpath(__file__))
        with tempfile.TemporaryDirectory(dir=test_directory, prefix=r"晚安_путь_к_файлу") as temp_dir:
            shape = [2, 3, 4]
            pd_model = make_pd_hapi_graph_model(shape)
            model_ref = make_ref_graph_model(shape)
            model_path = temp_dir + os.sep + 'model'

            try:
                pd_model.save(model_path, False)
            except:
                return
            model_path = model_path + ".pdmodel"

            assert os.path.exists(model_path), "Could not create a directory with unicode path."

            from openvino import convert_model, save_model, Core
            res_model = convert_model(model_path)
            flag, msg = compare_functions(res_model, model_ref, False)
            assert flag, msg

            save_model(res_model, model_path + ".xml")
            res_model_after_saving = Core().read_model(model_path + ".xml")
            flag, msg = compare_functions(res_model_after_saving, model_ref, False)
            assert flag, msg

            from openvino.frontend import FrontEndManager
            fm = FrontEndManager()
            fe = fm.load_by_framework("paddle")

            assert fe.supported(model_path)

            del res_model_after_saving

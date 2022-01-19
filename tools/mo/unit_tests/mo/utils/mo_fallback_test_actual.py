# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock
from unittest.mock import patch

import openvino
from openvino.tools.mo.main import prepare_ir
from openvino.frontend import FrontEndManager # pylint: disable=no-name-in-module,import-error
from onnx.helper import make_graph, make_model, make_tensor_value_info
import argparse
import os
import onnx
import paddle
import numpy as np
import shutil
import pytest
from generator import generator, generate

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

def base_args_config(use_legacy_fe:bool=None, use_new_fe:bool=None):
    args = argparse.Namespace()
    args.feManager = FrontEndManager()
    args.extensions = None
    args.use_legacy_frontend = use_legacy_fe
    args.use_new_frontend = use_new_fe
    args.framework = 'onnx'
    args.model_name = None
    args.input_model = None
    args.silent = True
    args.transform=[]
    args.legacy_ir_generation = False
    args.scale = None
    args.output=None
    args.input=None
    args.input_shape=None
    args.batch=None
    args.mean_values=None
    args.scale_values=None
    args.output_dir=os.getcwd()
    args.freeze_placeholder_with_value = None
    args.transformations_config = None
    args.disable_fusing = None
    args.finegrain_fusing = None
    args.disable_gfusing = None
    args.disable_resnet_optimization = None
    args.enable_concat_optimization = None
    args.static_shape = None
    args.disable_weights_compression = None
    args.reverse_input_channels = None
    args.data_type = None
    args.layout = None
    args.source_layout = None
    args.target_layout = None
    return args


def get_test_default_frontends():
    return {
        'onnx': 'new',
        'tf': 'legacy'
    }


def save_paddle_model(name, exe, feedkeys:list, fetchlist:list, target_dir:str):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    paddle.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe)
    paddle.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename=name+".pdmodel", params_filename=name+".pdiparams")


@generator
class TestMoFallback(unittest.TestCase):
    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()

        self.models = {}
        add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["add_out"])
        input_tensors = [
            make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (2, 2)),
            make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (2, 2)),
        ]
        output_tensors = [
            make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (1, 2)),
        ]
        graph = make_graph([add], "test_graph", input_tensors, output_tensors)
        model = make_model(graph, producer_name="MO tests",
                                                opset_imports=[onnx.helper.make_opsetid("", 13)])
        self.models["test_model.onnx"] = model

        for name, model in self.models.items():
            onnx.save(model, name)

        trans_config = 'config.json'
        with open(trans_config, 'w') as f:
            f.write("[]") # json format
        self.trans_config_file = os.path.abspath(trans_config)

        self.paddle_dir = "paddle_dir"
        paddle.enable_static()
        if not os.path.exists(self.paddle_dir):
            os.mkdir(self.paddle_dir)
        x = np.array([-2, 0, 1]).astype('float32')
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        out = paddle.nn.functional.relu(node_x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(paddle.static.default_startup_program())

        save_paddle_model("relu", exe, feedkeys=['x'], fetchlist=[out], target_dir=self.paddle_dir)


    def tearDown(self):
        for name in self.models.keys():
            os.remove(name)
        os.remove(self.trans_config_file)
        shutil.rmtree(self.paddle_dir)


    @generate(*[('dir_to_extension', None, None, 'mo_legacy', 'extensions'), # fallback
                ('dir_to_extension', None, True, 'onnx_frontend', None),
                ('dir_to_extension', True, None, 'mo_legacy', None),
                ('', True, None, 'mo_legacy', None),
                ('', None, True, 'onnx_frontend', None),
                (None, None, None, 'onnx_frontend', None),
    ])
    def test_fallback_if_extension_specified(self, extension, use_legacy, use_new_fe, conversion_method, fallback_reason):
        with patch('openvino.tools.mo.main.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.extensions = extension
            args.input_model = "test_model.onnx"
            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', conversion_method)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)


    @generate(*[(True, None, None, 'mo_legacy', 'transformations_config'), # fallback
                (True, True, None, 'mo_legacy', None),
                (False, None, True, 'onnx_frontend', None),
                (False, None, None, 'onnx_frontend', None),
    ])
    def test_fallback_if_tranformations_config_specified(self, trans_config_used, use_legacy, use_new_fe, expected_path, fallback_reason):
        with patch('openvino.tools.mo.main.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.input_model = "test_model.onnx"
            args.transformations_config = self.trans_config_file if trans_config_used else None

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)


    @generate(*[('dir_to_extension', True, None, 'mo_legacy', 'extensions, transformations_config'), # fallback
                (None, True, None, 'mo_legacy', 'transformations_config'), # fallback
                ('dir_to_extension', False, None, 'mo_legacy', 'extensions'), # fallback
                (None, False, True, 'onnx_frontend', None),
    ])
    def test_fallback_if_both_extension_and_trans_config_specified(self, extension, trans_config_used, use_new_fe, expected_path, fallback_reason):
        with patch('openvino.tools.mo.main.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_new_fe=use_new_fe)
            args.extensions = extension
            args.input_model = "test_model.onnx"
            args.transformations_config = self.trans_config_file if trans_config_used else None

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)


    @generate(*[(True, None, None, 'mo_legacy'),
                (True, True, None, 'mo_legacy'),
                (False, None, True, 'onnx_frontend'),
    ])
    def test_fallback_if_legacy_set_as_default(self, trans_config_used, use_legacy, use_new_fe, expected_path):
        with patch('openvino.tools.mo.main.get_default_frontends') as default_fe:
            default_fe.return_value = {'onnx': 'legacy', 'tf': 'legacy'}
            args = base_args_config(use_legacy, use_new_fe)
            args.input_model = "test_model.onnx"
            args.transformations_config = self.trans_config_file if trans_config_used else None

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            with pytest.raises(AssertionError): # not called
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason')


    @generate(*[(None, None, 'dir_to_extension', 'paddle_frontend'),
                (True, None, None, 'paddle_frontend'),
                (None, None, None, 'paddle_frontend'),
    ])
    def test_no_fallback_if_pdpd(self, use_new_fe, use_legacy, extension, expected_path):
        args = base_args_config(use_legacy, use_new_fe)
        args.framework = 'paddle'
        args.extensions = extension
        args.input_model = 'paddle_dir/relu/relu.pdmodel'

        prepare_ir(args)

        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
        with pytest.raises(AssertionError): # not called
            tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason')

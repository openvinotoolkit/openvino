# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
from unittest.mock import patch, Mock

import numpy as np
import onnx
import pytest
from onnx.helper import make_graph, make_model, make_tensor_value_info
from openvino.frontend import FrontEndManager, FrontEnd  # pylint: disable=no-name-in-module,import-error

from openvino.tools.mo.convert_impl import prepare_ir
from openvino.tools.mo.utils.error import Error

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

try:
    import paddle
    paddle_imported = True
except ImportError:
    paddle_imported = False


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
    args.static_shape = None
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


class TestMoFallback():
    def setup_method(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()
        FrontEnd.add_extension = Mock()

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

        self.test_config_files = {}
        self.test_config_files['fake_config.json'] = '[]' # json format

        self.test_config_files['test_config.json'] = """[
            {
            "custom_attributes": {
            "test_attribute": true
            },
            "id": "TransformationName1",
            "match_kind": "scope"
            },
            { 
            "custom_attributes": {
            },
            "id": "TransfromationName2",
            "match_kind": "scope"
            }
        ]"""

        self.test_config_files['onnx_fe_ext.so'] = 'binary_content'
        self.test_config_files['onnx_fe_ext_2.so'] = 'binary_content'

        for file, content in self.test_config_files.items():
            with open(file, 'w') as f:
                f.write(content)

        if paddle_imported:
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

    def teardown_method(self):
        for name in self.models.keys():
            os.remove(name)
        for name in self.test_config_files:
            os.remove(name)
        if paddle_imported:
            shutil.rmtree(self.paddle_dir)

    @pytest.mark.parametrize("extension, use_legacy, use_new_fe, conversion_method, fallback_reason", [
        (['dir_to_extension'], None, None, 'mo_legacy', 'extensions'),  # fallback
        (['dir_to_extension'], None, True, None, None),  # exception
        (['dir_to_extension'], True, None, 'mo_legacy', None),
        ([''], True, None, 'mo_legacy', None),
        ([''], None, True, 'onnx_frontend', None),
        (None, None, None, 'onnx_frontend', None)
    ])
    def test_fallback_if_extension_specified(self, extension, use_legacy, use_new_fe, conversion_method, fallback_reason):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.extensions = extension
            args.input_model = "test_model.onnx"

            if conversion_method:
                prepare_ir(args)
                tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', conversion_method)
                if fallback_reason:
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
                else:
                    with pytest.raises(AssertionError): # not called
                        tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(Error): # not supported extensions on new path
                    prepare_ir(args)

    @pytest.mark.parametrize("use_legacy, use_new_fe, conversion_method", [
        (None, None, 'onnx_frontend'),
        (True, None, None),  # exception
        (None, True, 'onnx_frontend'),
    ])
    def test_fallback_if_new_extension_specified(self, use_legacy, use_new_fe, conversion_method):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.extensions = ['onnx_fe_ext.so']
            args.input_model = "test_model.onnx"

            if conversion_method:
                prepare_ir(args)
                tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', conversion_method)
            else:
                with pytest.raises(Error):
                    prepare_ir(args)

    @pytest.mark.parametrize("use_legacy, use_new_fe, conversion_method", [
        (None, None, 'onnx_frontend'),
        (True, None, None),  # exception
        (None, True, 'onnx_frontend')
    ])
    def test_fallback_if_two_new_extension_specified(self, use_legacy, use_new_fe, conversion_method):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.extensions = ['onnx_fe_ext.so', 'onnx_fe_ext_2.so']
            args.input_model = "test_model.onnx"

            if conversion_method:
                prepare_ir(args)
                tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', conversion_method)
            else:
                with pytest.raises(Error):
                    prepare_ir(args)

    @pytest.mark.parametrize("trans_config, use_legacy, use_new_fe, expected_path, fallback_reason", [
        ('fake_config.json', None, None, 'mo_legacy', 'transformations_config'),  # fallback
        ('test_config.json', None, None, 'mo_legacy', 'transformations_config'),  # fallback
        ('fake_config.json', True, None, 'mo_legacy', None),
        (None, None, True, 'onnx_frontend', None),
        (None, None, None, 'onnx_frontend', None)])
    def test_fallback_if_tranformations_config_specified(self, trans_config, use_legacy, use_new_fe, expected_path,
                                                         fallback_reason):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_legacy, use_new_fe)
            args.input_model = "test_model.onnx"
            args.transformations_config = trans_config

            with patch('openvino.tools.mo.utils.class_registration.apply_transform'): # skip applying transforms
                prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)

    @pytest.mark.parametrize("extension, trans_config, use_new_fe, expected_path, fallback_reason", [
        (['dir_to_extension'], 'fake_config.json', None, 'mo_legacy', 'extensions, transformations_config'),  # fallback
        (None, 'fake_config.json', None, 'mo_legacy', 'transformations_config'),  # fallback
        (['dir_to_extension'], None, None, 'mo_legacy', 'extensions'),  # fallback
        (None, None, True, 'onnx_frontend', None)
    ])
    def test_fallback_if_both_extension_and_trans_config_specified(self, extension, trans_config, use_new_fe, expected_path, fallback_reason):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = get_test_default_frontends()
            args = base_args_config(use_new_fe=use_new_fe)
            args.extensions = extension
            args.input_model = "test_model.onnx"
            args.transformations_config = trans_config

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)

    @pytest.mark.parametrize("trans_config, use_legacy, use_new_fe, expected_path",
                             [('fake_config.json', None, None, 'mo_legacy'),
                              ('fake_config.json', True, None, 'mo_legacy'),
                              (None, None, True, 'onnx_frontend')])
    def test_fallback_if_legacy_set_as_default(self, trans_config, use_legacy, use_new_fe, expected_path):
        with patch('openvino.tools.mo.convert_impl.get_default_frontends') as default_fe:
            default_fe.return_value = {'onnx': 'legacy', 'tf': 'legacy'}
            args = base_args_config(use_legacy, use_new_fe)
            args.input_model = "test_model.onnx"
            args.transformations_config = trans_config

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
            with pytest.raises(AssertionError): # not called
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason')

    @pytest.mark.skipif(not paddle_imported, reason="PaddlePaddle is not installed")
    @pytest.mark.parametrize("use_new_fe, use_legacy, extension, expected_path",
                             [(True, None, None, 'paddle_frontend'),
                              (None, None, None, 'paddle_frontend')])
    def test_no_fallback_if_pdpd(self, use_new_fe, use_legacy, extension, expected_path):
        args = base_args_config(use_legacy, use_new_fe)
        args.framework = 'paddle'
        args.extensions = extension
        args.input_model = 'paddle_dir/relu/relu.pdmodel'

        prepare_ir(args)

        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
        with pytest.raises(AssertionError): # not called
            tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason')

    @pytest.mark.skipif(not paddle_imported, reason="PaddlePaddle is not installed")
    def test_exception_if_old_extensions_used_for_pdpd(self):
        args = base_args_config()
        args.framework = 'paddle'
        args.extensions = ['dir_to_extension']
        args.input_model = 'paddle_dir/relu/relu.pdmodel'

        with pytest.raises(Error) as ex: # not called
            prepare_ir(args)
            assert str(ex) == 'Legacy transformations configuration is not supported for the new frontend'

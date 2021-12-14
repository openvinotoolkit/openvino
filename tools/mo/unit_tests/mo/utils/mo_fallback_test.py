# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock
from unittest.mock import patch
import pytest

from openvino.tools.mo.main import prepare_ir
from openvino.frontend import FrontEndManager # pylint: disable=no-name-in-module,import-error
from onnx.helper import make_graph, make_model, make_tensor_value_info
import argparse
import os
import onnx
from generator import generator, generate
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.class_registration import _update
from openvino.tools.mo.ops.activation_ops import Elu

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm

def base_args_config():
    args = argparse.Namespace()
    args.feManager = FrontEndManager()
    args.extensions = None
    args.use_legacy_frontend = False
    args.use_new_frontend = True
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


def event_call_args_to_dict(call_args):
    result = {}
    for call in call_args:
        _, name, value, *_ = call.args # skip 'mo' and unused event param
        result[name] = value
    return result


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

        fake_squeeze = onnx.helper.make_node(
            'FakeElu',
            inputs=['x'],
            outputs=['y'],
            axes=[0],
        )
        input_tensors = [
            make_tensor_value_info("x", onnx.TensorProto.FLOAT, (2, 2)),
        ]
        output_tensors = [
            make_tensor_value_info("y", onnx.TensorProto.FLOAT, (2, 2)),
        ]
        graph = make_graph([fake_squeeze], "test_graph", input_tensors, output_tensors)
        model = make_model(graph, producer_name="MO tests",
                                                opset_imports=[onnx.helper.make_opsetid("", 13)])
        self.models["fake_elu.onnx"] = model

        for name, model in self.models.items():
            onnx.save(model, name)

    def tearDown(self):
        for name in self.models.keys():
            os.remove(name)


    @generate(*[('dir_to_extension', False, True, 'mo_legacy', None),
                ('dir_to_extension', True, False, 'mo_legacy', 'extensions'),
                ('', True, False, 'mo_legacy', None),
                ('', False, True, 'onnx_frontend', None),
                (None, False, True, 'onnx_frontend', None),
    ])
    def test_fallback_if_extension_specified(self, extension, use_legacy, use_new_fe, conversion_method, fallback_reason):
        # fix problem with incorrect extractors loading from UT
        with patch('openvino.tools.mo.utils.class_registration._update') as update_mock:
            update_mock.return_value=None
            args = base_args_config()
            args.extensions = extension
            args.use_legacy_frontend = use_legacy
            args.use_new_frontend = use_new_fe
            args.input_model = "test_model.onnx"

            prepare_ir(args)

            tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', conversion_method)
            if fallback_reason:
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
            else:
                with pytest.raises(AssertionError): # not called
                    tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)



    @generate(*[('config.json', False, True, 'mo_legacy', None),
                ('config.json', True, False, 'mo_legacy', 'transformations_config'),
                (None, False, True, 'onnx_frontend', None),
    ])
    def test_fallback_if_tranformation_config_specified(self, trans_config, use_legacy, use_new_fe, expected_path, fallback_reason):
        args = base_args_config()
        args.use_legacy_frontend = use_legacy
        args.use_new_frontend = use_new_fe
        args.input_model = "test_model.onnx"
        if trans_config is not None: # trans config provided
            with open(trans_config, 'w') as f:
                f.write("[]") # json format
            args.transformations_config = os.path.abspath(trans_config)
        else:
            args.transformations_config = trans_config

        prepare_ir(args)

        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
        if fallback_reason:
            tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
        else:
            with pytest.raises(AssertionError): # not called
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)

        if args.transformations_config is not None:
            os.remove(args.transformations_config) # clean-up


    @generate(*[('dir_to_extension', 'config.json', True, 'mo_legacy', 'extensions, transformations_config'),
                (None, 'config.json', True, 'mo_legacy', 'transformations_config'),
                ('dir_to_extension', None, True, 'mo_legacy', 'extensions'),
                (None, None, True, 'onnx_frontend', None),
    ])
    def test_fallback_if_both_extension_and_trans_config_specified(self, extension, trans_config, use_new_fe, expected_path, fallback_reason):
        args = base_args_config()
        args.use_new_frontend = use_new_fe
        args.extensions = extension
        args.input_model = "test_model.onnx"
        if trans_config is not None: # trans config provided
            with open(trans_config, 'w') as f:
                f.write("[]") # json format
            args.transformations_config = os.path.abspath(trans_config)
        else:
            args.transformations_config = trans_config

        prepare_ir(args)
        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
        if fallback_reason:
            tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)
        else:
            with pytest.raises(AssertionError): # not called
                tm.Telemetry.send_event.assert_any_call('mo', 'fallback_reason', fallback_reason)

        if args.transformations_config is not None:
            os.remove(args.transformations_config) # clean-up


    @generate(*[(True, False, 'mo_legacy', 'nGraph does not support the following ONNX operations: FakeElu'),
                (False, False, 'mo_legacy', None),
    ])
    def test_fallback_if_frontend_path_failed(self, use_new_fe, use_legacy, expected_path, fallback_reason):
        args = base_args_config()
        args.use_legacy_frontend = use_legacy
        args.use_new_frontend = use_new_fe
        args.input_model = "fake_elu.onnx"
        # FakeElu is supported only by legacy path
        Op.registered_ops['FakeElu'] = Elu

        prepare_ir(args)

        call_args_dict = event_call_args_to_dict(tm.Telemetry.send_event.call_args_list)
        assert call_args_dict['conversion_method'] == expected_path
        if fallback_reason is None:
            assert not 'fallback_reason' in call_args_dict
        else:
            assert fallback_reason in call_args_dict['fallback_reason'] # do not check callstack in a exception

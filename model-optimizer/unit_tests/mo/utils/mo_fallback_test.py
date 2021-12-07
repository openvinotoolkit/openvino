# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

from mo.main import prepare_ir
from openvino.frontend import FrontEndManager # pylint: disable=no-name-in-module,import-error
from onnx.helper import make_graph, make_model, make_tensor_value_info
import argparse
import os
import onnx
from generator import generator, generate
from mo.ops.op import Op
from extensions.ops.activation_ops import Elu

try:
    import openvino_telemetry as tm
except ImportError:
    import mo.utils.telemetry_stub as tm

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
    return args


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


    @generate(*[('dir_to_extension', False, True, 'mo_legacy'),
                ('dir_to_extension', True, False, 'mo_legacy'),
                ('', True, False, 'mo_legacy'),
                ('', False, True, 'onnx_frontend'),
                (None, False, True, 'onnx_frontend'),
    ])
    def test_fallback_if_extension_specified(self, extension, use_legacy, use_new_fe, expected_path):
        args = base_args_config()
        args.extensions = extension
        args.use_legacy_frontend = use_legacy
        args.use_new_frontend = use_new_fe
        args.input_model = "test_model.onnx"

        prepare_ir(args)
        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)

    @generate(*[('config.json', False, True, 'mo_legacy'),
                ('config.json', True, False, 'mo_legacy'),
                (None, False, True, 'onnx_frontend'),
    ])
    def test_fallback_if_tranformation_config_specified(self, trans_config, use_legacy, use_new_fe, expected_path):
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

        if args.transformations_config is not None:
            os.remove(args.transformations_config) # clean-up


    @generate(*[('dir_to_extension', 'config.json', True, 'mo_legacy'),
                (None, 'config.json', True, 'mo_legacy'),
                ('dir_to_extension', None, True, 'mo_legacy'),
                (None, None, True, 'onnx_frontend'),
    ])
    def test_fallback_if_both_extension_and_trans_config_specified(self, extension, trans_config, use_new_fe, expected_path):
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

        if args.transformations_config is not None:
            os.remove(args.transformations_config) # clean-up


    @generate(*[(True, False, 'mo_legacy'),
                (False, False, 'mo_legacy'),
    ])
    def test_fallback_if_frontend_path_failed(self, use_new_fe, use_legacy, expected_path):
        args = base_args_config()
        args.use_legacy_frontend = True
        args.extensions='ext_path'
        args.use_new_frontend = False
        args.input_model = "fake_elu.onnx"
        # FakeElu is supported only by legacy path
        Op.registered_ops['FakeElu'] = Elu

        graph, ng_func = prepare_ir(args)

        tm.Telemetry.send_event.assert_any_call('mo', 'conversion_method', expected_path)
        assert graph
        assert not ng_func

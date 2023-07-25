# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch, Mock
import onnx
from onnx.helper import make_graph, make_model, make_tensor_value_info
import os
from os import environ
import json
import argparse
from openvino.tools.mo.convert_impl import prepare_ir
from openvino.frontend import FrontEndManager # pylint: disable=no-name-in-module,import-error


try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
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
    args.input_checkpoint = None
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
    args.frontend_defaults = {
        'onnx': 'legacy',
        'tf': 'legacy'
    }
    return args


class TestMoFallback(unittest.TestCase):
    def setUp(self):
        environ.update({'MO_ENABLED_TRANSFORMS': 'ANALYSIS_JSON_PRINT'})

        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()

        self.models = {}
        add = onnx.helper.make_node("Add", inputs=["in1", "in2"], outputs=["add_out"])
        input_tensors = [
            make_tensor_value_info("in1", onnx.TensorProto.FLOAT, (1, 2)),
            make_tensor_value_info("in2", onnx.TensorProto.FLOAT, (1, 2)),
        ]
        output_tensors = [
            make_tensor_value_info("add_out", onnx.TensorProto.FLOAT, (1, 2)),
        ]
        graph = make_graph([add], "test_graph", input_tensors, output_tensors)
        model = make_model(graph, producer_name="MO tests",
                                                opset_imports=[onnx.helper.make_opsetid("", 13)])
        self.models["test_model.onnx"] = model

        input_tensors_2 = [
            make_tensor_value_info("in1", onnx.TensorProto.INT64, (1, 'dyn_dim', 3)),
            make_tensor_value_info("in2", onnx.TensorProto.INT64, None),
            make_tensor_value_info("in3", onnx.TensorProto.INT64, ()),
        ]
        output_tensors_2 = [
            make_tensor_value_info("mul_out", onnx.TensorProto.FLOAT, None),
        ]
        mul = onnx.helper.make_node("Mul", inputs=["add_out", "in3"], outputs=["mul_out"])
        graph_2 = make_graph([add, mul], "test_graph_2", input_tensors_2, output_tensors_2)
        model_2 = make_model(graph_2, producer_name="MO tests",
                                                opset_imports=[onnx.helper.make_opsetid("", 13)])
        self.models["test_model_2.onnx"] = model_2

        split_1 = onnx.helper.make_node("Split", inputs=["add_out"],
                                  outputs=["out1", "out2"], axis=0)
        split_2 = onnx.helper.make_node("Split", inputs=["mul_out"],
                                  outputs=["out3", "out4"], axis=0)
        output_tensors_3 = [
            make_tensor_value_info("out1", onnx.TensorProto.FLOAT, 'dyn_dim'),
            make_tensor_value_info("out2", onnx.TensorProto.FLOAT, 'dyn_dim'),
            make_tensor_value_info("out3", onnx.TensorProto.FLOAT, 'dyn_dim'),
            make_tensor_value_info("out4", onnx.TensorProto.FLOAT, 'dyn_dim'),
        ]
        graph_3 = make_graph([add, mul, split_1, split_2], "test_graph_3", input_tensors_2, output_tensors_3)
        model_3 = make_model(graph_3, producer_name="MO tests",
                                                opset_imports=[onnx.helper.make_opsetid("", 13)])
        self.models["test_model_3.onnx"] = model_3

        for name, model in self.models.items():
            onnx.save(model, name)

    def tearDown(self):
        del environ['MO_ENABLED_TRANSFORMS']
        for name in self.models.keys():
            os.remove(name)


    @patch('openvino.tools.mo.moc_frontend.analysis.json_model_analysis_print')
    def test_model(self, json_print):
        args = base_args_config()
        args.input_model = "test_model.onnx"

        with patch('sys.exit') as exit_mock: # do not exit execution
            prepare_ir(args)

        result = json_print.call_args.args[0]

        assert 'inputs' in result
        assert result['inputs'] == json.loads('{"in1": {"shape": [1, 2], "data_type": "float32", "value": "None"}, \
                                                "in2": {"shape": [1, 2], "data_type": "float32", "value": "None"}}')

        assert 'intermediate' in result
        assert result['intermediate'] == json.loads('{"in1": {"shape": [1, 2], "data_type": "float32", "value": "None"}, \
                                                      "in2": {"shape": [1, 2], "data_type": "float32", "value": "None"}, \
                                                      "add_out": {"shape": "None", "data_type": "None", "value": "None"}}')


    @patch('openvino.tools.mo.moc_frontend.analysis.json_model_analysis_print')
    def test_model_with_dyn_shapes(self, json_print):
        args = base_args_config()
        args.input_model = "test_model_2.onnx"

        with patch('sys.exit') as exit_mock: # do not exit execution
            prepare_ir(args)

        result = json_print.call_args.args[0]

        assert 'inputs' in result
        print(result['inputs'])
        assert result['inputs'] == json.loads('{"in1": {"shape": [1, 0, 3], "data_type": "int64", "value": "None"}, \
                                                "in2": {"shape": "None", "data_type": "int64", "value": "None"}, \
                                                "in3": {"shape": [], "data_type": "int64", "value": "None"}}')

        assert 'intermediate' in result
        assert result['intermediate'] == json.loads('{"in1": {"shape": [1, 0, 3], "data_type": "int64", "value": "None"}, \
                                                      "in2": {"shape": "None", "data_type": "int64", "value": "None"}, \
                                                      "in3": {"shape": [], "data_type": "int64", "value": "None"}, \
                                                      "mul_out": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "add_out": {"shape": "None", "data_type": "None", "value": "None"}}')


    @patch('openvino.tools.mo.moc_frontend.analysis.json_model_analysis_print')
    def test_multi_outputs_model(self, json_print):
        args = base_args_config()
        args.input_model = "test_model_3.onnx"

        with patch('sys.exit') as exit_mock: # do not exit execution
            prepare_ir(args)

        result = json_print.call_args.args[0]

        assert 'inputs' in result
        assert result['inputs'] == json.loads('{"in1": {"shape": [1, 0, 3], "data_type": "int64", "value": "None"}, \
                                                "in2": {"shape": "None", "data_type": "int64", "value": "None"}, \
                                                "in3": {"shape": [], "data_type": "int64", "value": "None"}}')

        assert 'intermediate' in result
        assert result['intermediate'] == json.loads('{"in1": {"shape": [1, 0, 3], "data_type": "int64", "value": "None"}, \
                                                      "in2": {"shape": "None", "data_type": "int64", "value": "None"}, \
                                                      "in3": {"shape": [], "data_type": "int64", "value": "None"}, \
                                                      "mul_out": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "add_out": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "out1": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "out2": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "out3": {"shape": "None", "data_type": "None", "value": "None"}, \
                                                      "out4": {"shape": "None", "data_type": "None", "value": "None"}}')

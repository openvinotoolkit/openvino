# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch
import onnx
from onnx.helper import make_graph, make_model, make_tensor_value_info
import os
import json
from openvino.tools.mo.main import prepare_ir
from openvino.frontend import FrontEndManager # pylint: disable=no-name-in-module,import-error
import argparse
from openvino.tools.mo.analysis.json_print_new_frontend import json_model_analysis_print

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
    args.frontend_defaults = {
        'onnx': 'legacy',
        'tf': 'legacy'
    }
    return args


class TestMoFallback(unittest.TestCase):
    def setUp(self):
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

        for name, model in self.models.items():
            onnx.save(model, name)

    def tearDown(self):
        for name in self.models.keys():
            pass
            #os.remove(name)

    @patch('openvino.tools.mo.analysis.json_print_new_frontend.json_model_analysis_print')
    def test_model(self, json_print):
        args = base_args_config()
        args.input_model = "test_model.onnx"

        prepare_ir(args)

        expected = json.loads('{"intermediate": {"in1": {"shape": [1, 2], "data_type": "None", "value": "None"}, '
        '"in2": {"shape": [1, 2], "data_type": "None", "value": "None"}, "add_out": {"shape": "None", "data_type": "None", "value": "None"}, '
        '"add_out4": {"shape": "None", "data_type": "None", "value": "None"}}, "inputs": {"in1": {"shape": [1, 2], "data_type": "float32", "value": "None"}, '
        '"in2": {"shape": [1, 2], "data_type": "float32", "value": "None"}}}')

        result = json_print.call_args.args[0]

        assert 'inputs' in result
        print(result)
        assert result['inputs'] == expected['inputs']

# test unknown shape
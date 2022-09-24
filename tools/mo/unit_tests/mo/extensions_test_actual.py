# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import unittest
from unittest.mock import Mock
import onnx
from onnx.helper import make_graph, make_model, make_tensor_value_info
import os
from os import path
import json
import argparse
from pathlib import Path
from itertools import chain
from openvino.tools.mo.convert_impl import prepare_ir
from openvino.frontend import (
    FrontEndManager,
)  # pylint: disable=no-name-in-module,import-error

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
    args.framework = "onnx"
    args.model_name = None
    args.input_model = None
    args.silent = True
    args.transform = []
    args.legacy_ir_generation = False
    args.scale = None
    args.output = None
    args.input = None
    args.input_shape = None
    args.batch = None
    args.mean_values = None
    args.scale_values = None
    args.output_dir = os.getcwd()
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


def get_builtin_extensions_path():
    win_folder_path = Path(__file__).parent.parent.parent.parent
    linux_folder_path = win_folder_path.joinpath("lib")
    for lib_path in chain(
        win_folder_path.glob("*.dll"), linux_folder_path.glob("*.so")
    ):
        if "libtest_builtin_extensions_1" in lib_path.name:
            return str(lib_path)
    return ""


class TestMoFallback(unittest.TestCase):
    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()

        self.models = {}
        relu = onnx.helper.make_node("Relu", inputs=["in"], outputs=["out"])
        input_tensors = [
            make_tensor_value_info("in", onnx.TensorProto.FLOAT, (1, 2)),
        ]
        output_tensors = [
            make_tensor_value_info("out", onnx.TensorProto.FLOAT, (1, 2)),
        ]
        graph = make_graph([relu], "test_graph", input_tensors, output_tensors)
        model = make_model(
            graph,
            producer_name="MO tests",
            opset_imports=[onnx.helper.make_opsetid("", 13)],
        )
        self.models["test_model.onnx"] = model

        self.test_config_files = {}
        self.test_config_files[
            "test_config.json"
        ] = """[
            {
            "custom_attributes": {
            "test_attribute": true
            },
            "id": "buildin_extensions_1::TestExtension1",
            "library": "library_path",
            "match_kind": "scope"
            }
        ]""".replace(
            "library_path", get_builtin_extensions_path()
        )

        for name, model in self.models.items():
            onnx.save(model, name)
        for name, config in self.test_config_files.items():
            with open(name, "w") as f:
                f.write(config)

    def tearDown(self):
        for name in self.models.keys():
            os.remove(name)
        for name in self.test_config_files.keys():
            os.remove(name)

    @pytest.mark.skipif(
        len(get_builtin_extensions_path()) == 0,
        reason="The extension library path was not found",
    )
    def test_conersion_if_extensions_is_used(self):
        args = base_args_config()
        args.input_model = "test_model.onnx"
        args.extensions = [get_builtin_extensions_path()]

        graph, model = prepare_ir(args)

        assert any(op.get_type_name() == "Swish" for op in model.get_ops())
        assert all(op.get_type_name() != "Relu" for op in model.get_ops())

    @pytest.mark.skipif(
        len(get_builtin_extensions_path()) == 0,
        reason="The extension library path was not found",
    )
    def test_conersion_if_transformations_config_is_used(self):
        args = base_args_config()
        args.input_model = "test_model.onnx"
        args.transformations_config = "test_config.json"

        graph, model = prepare_ir(args)

        assert model.get_friendly_name() == "TestFunction"

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

import onnx
from onnx import helper, TensorProto


rng = np.random.default_rng()


class TestLabelEncoder(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_info):
        assert "input" in inputs_info
        input_shape = inputs_info["input"]
        inputs_data = {}
        sample_data = rng.choice(self.strings_dictionary, input_shape)
        inputs_data["input"] = sample_data
        return inputs_data

    def create_net(self, shape, ir_version, strings_dictionary):
        """
        ONNX net / IR net

        Input -> LabelEncoder -> Output
        """

        # Create ONNX model
        self.strings_dictionary = strings_dictionary

        input = helper.make_tensor_value_info("input", TensorProto.STRING, shape)
        output = helper.make_tensor_value_info("output", TensorProto.INT64, shape)

        node_def = onnx.helper.make_node(
            "LabelEncoder",
            inputs=["input"],
            outputs=["output"],
            domain="ai.onnx.ml",  # LabelEncoder belongs to the ML domain
            keys_strings=[
                "16384",
                "16385",
                "16386",
                "16387",
                "16388",
                "16389",
                "16390",
                "16392",
                "16393",
                "16394",
                "16397",
                "16398",
                "16400",
                "16401",
                "16402",
                "16403",
                "16404",
                "16407",
                "16409",
                "16410",
                "16411",
                "16413",
                "16419",
                "16425",
                "16426",
                "16427",
                "16430",
                "16432",
                "16433",
                "16434",
                "16435",
                "16456",
                "16457",
                "16461",
                "16462",
                "16468",
                "16469",
                "16470",
                "16471",
                "16472",
                "16481",
                "16490",
                "16491",
                "16494",
                "16495",
                "16496",
                "16497",
                "16505",
                "16522",
                "16523",
                "16524",
                "16525",
                "16526",
                "16530",
                "16534",
                "16535",
                "16538",
                "61440",
            ],  # Input string labels
            values_int64s=list(range(58)),
            default_int64=-1,  # Default value if label not found
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            "test_model",
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name="test_model")
        onnx_net.ir_version = 8

        # Create reference IR net (obsolete code)
        ref_net = None
        return onnx_net, ref_net

    @pytest.mark.parametrize("input_shape", [[3]])
    @pytest.mark.parametrize("strings_dictionary", [["16385", "16386", "16387"]])
    @pytest.mark.nightly
    def test_labelencoder(
        self,
        input_shape,
        strings_dictionary,
        ie_device,
        precision,
        ir_version,
        temp_dir,
    ):
        self._test(
            *self.create_net(
                shape=input_shape,
                strings_dictionary=strings_dictionary,
                ir_version=ir_version,
            ),
            ie_device,
            precision,
            ir_version,
            temp_dir=temp_dir,
        )

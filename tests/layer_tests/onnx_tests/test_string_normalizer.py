# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

import onnx
from onnx import helper, TensorProto

rng = np.random.default_rng()


class TestStringNormalizer(OnnxRuntimeLayerTest):

    def _prepare_input(self, inputs_info):
        assert "input" in inputs_info
        input_shape = inputs_info["input"]
        sample_data = rng.choice(self.strings_dictionary, input_shape)
        return {"input": sample_data}

    def create_net(self, shape, ir_version, strings_dictionary):
        """
        ONNX net IR net

        Input->StringNormalizer->Output => Input->RegexNormalization->CaseFold
        """
        # Store dictionary for use in _prepare_input
        self.strings_dictionary = strings_dictionary

        # Create ONNX model
        input_info = helper.make_tensor_value_info("input", TensorProto.STRING, shape)
        output_info = helper.make_tensor_value_info("output", TensorProto.STRING, shape)

        node_def = onnx.helper.make_node(
            "StringNormalizer",
            inputs=["input"],
            outputs=["output"],
            case_change_action="LOWER",  # Options: "NONE" | "UPPER" | "LOWER"
            locale="en_US",  # Locale for normalization
            is_case_sensitive=0,
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            "test_model",
            [input_info],
            [output_info],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name="test_model")
        onnx_net.ir_version = 8

        # Create reference IR net (obsolete code)
        ref_net = None

        return onnx_net, ref_net

    @pytest.mark.parametrize("input_shape", [[2]])
    @pytest.mark.parametrize(
        "strings_dictionary",
        [
            [
                "UPPER CASE SENTENCE",
                "lower case sentence",
                " UppEr LoweR CAse SENtence",
                "BCD EFG",
            ]
        ],
    )
    @pytest.mark.nightly
    def test_stringnormalizer(
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

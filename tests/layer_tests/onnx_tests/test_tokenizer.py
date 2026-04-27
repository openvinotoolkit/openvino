# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model
import numpy as np


rng = np.random.default_rng()


class TestTokenizer(OnnxRuntimeLayerTest):

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

        Input->TestTokenizer->Output => Input->RegexSplit
        """

        #
        # Create ONNX model
        #
        import onnx
        from onnx import helper
        from onnx import TensorProto

        self.strings_dictionary = strings_dictionary

        input = helper.make_tensor_value_info("input", TensorProto.STRING, shape)
        output = helper.make_tensor_value_info("output", TensorProto.STRING, shape)

        node_def = onnx.helper.make_node(
            op_type="Tokenizer",
            inputs=["input"],
            outputs=["output"],
            domain="com.microsoft",
            name="Tokenizer_0",
            # tokenexp set: regex pattern for tokenization
            tokenexp="[a-zA-Z0-9_]+",
            # Do NOT add begin / end markers
            mark=False,
            # Minimum character count per token
            mincharnum=1,
            # Padding value for output
            pad_value="#",
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            "test_model",
            [input],
            [output],
        )

        # Create the model (ModelProto)
        # Standard ONNX opset
        onnx_opset = helper.make_opsetid("", 18)
        ms_opset = helper.make_opsetid("com.microsoft", 1)
        opset_imports = [onnx_opset, ms_opset]
        onnx_net = onnx_make_model(
            graph_def,
            opset_imports=opset_imports,
            producer_name="test_model",
        )
        onnx_net.ir_version = 8

        #
        # Create reference IR net (obsolete code)
        #
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
    def test_tokenizer(
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

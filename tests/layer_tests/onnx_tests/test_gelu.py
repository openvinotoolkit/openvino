# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest

from unit_tests.utils.graph import build_graph


class TestGelu(OnnxRuntimeLayerTest):
    def create_net(self, shape, ir_version):
        """
            ONNX net                   IR net

            Input->Gelu->Output   =>    Input->gelu

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)
     

        node_def = onnx.helper.make_node(
            'Gelu',
            inputs=['input'],
            outputs=['output'],
            domain='com.microsoft'
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = helper.make_model(graph_def, producer_name='test_model')

        #   Create reference IR net
        ref_net = None

        return onnx_net, ref_net


    test_data = [dict(shape=[10, 12]),
                 dict(shape=[8, 10, 12]),
                 dict(shape=[6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_gelu(self, params, ie_device, precision, ir_version, temp_dir, api_2):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir, api_2=api_2)


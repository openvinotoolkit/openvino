# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model


class TestMeanVarianceNormalization(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shape, axes, ir_version):
        """
            ONNX net                                 IR net

            Input->MeanVarianceNormalization->Output   =>    Input->MVN
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
            'MeanVarianceNormalization',
            inputs=['input'],
            outputs=['output'],
            axes=axes
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output]
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return onnx_net, ref_net

    test_data = [
        dict(shape=[7, 2, 3, 5], axes=[2, 3]),
        dict(shape=[7, 2, 3, 5], axes=[1, 2, 3]),
        dict(shape=[7, 2, 3, 5, 11], axes=[2, 3, 4]),
        dict(shape=[7, 2, 3, 5, 11], axes=[1, 2, 3, 4])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_mvn(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

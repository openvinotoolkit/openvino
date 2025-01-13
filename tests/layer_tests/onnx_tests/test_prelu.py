# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestPRelu(OnnxRuntimeLayerTest):
    def create_net(self, shape, slope_shape, precision, ir_version, opset=None):
        """
            ONNX net                     IR net

            Input->PRelu->Output   =>    Input->PReLU

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        const = np.random.randn(*slope_shape).astype(np.float32)

        node_slope_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['slope'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=const.shape,
                vals=const.flatten(),
            ),
        )

        node_def = onnx.helper.make_node(
            'PRelu',
            inputs=['input', 'slope'],
            outputs=['output']
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_slope_def, node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        args = dict(producer_name='test_model')
        if opset:
            args['opset_imports'] = [helper.make_opsetid("", opset)]
        onnx_net = onnx_make_model(graph_def, **args)

        #
        #   Create reference IR net
        #
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'weights_indata': {'kind': 'data', 'value': const.flatten()},
                'weights': {'kind': 'op', 'type': 'Const'},
                'weights_data': {'kind': 'data', 'shape': [len(const.flatten())]},
                'node': {'kind': 'op', 'type': 'PReLU'},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('weights_indata', 'weights'),
                                   ('weights', 'weights_data'),
                                   ('weights_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return onnx_net, ref_net

    # Note: OV only support slopes of one element or of size equal to number of channels.
    test_data_shared_channels = [
        dict(shape=[10, 12], slope_shape=[12]),
        dict(shape=[8, 10, 12], slope_shape=[10, 1]),
        dict(shape=[6, 8, 10, 12], slope_shape=[8, 1, 1]),
        dict(shape=[4, 6, 8, 10, 12], slope_shape=[6, 1, 1, 1])]

    test_data_scalar_precommit = [
        dict(shape=[2, 4, 6, 8], slope_shape=[1]),
        dict(shape=[2, 4, 6, 8, 10], slope_shape=[1])
    ]

    test_data_scalar = [
        dict(shape=[10, 12], slope_shape=[1]),
        dict(shape=[8, 10, 12], slope_shape=[1]),
        dict(shape=[6, 8, 10, 12], slope_shape=[1]),
        dict(shape=[4, 6, 8, 10, 12], slope_shape=[1])]

    test_data_precommit = [dict(shape=[8, 10, 12], slope_shape=[12])]

    @pytest.mark.parametrize("params", test_data_scalar)
    @pytest.mark.nightly
    def test_prelu_opset7_scalar(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, opset=7, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_shared_channels)
    @pytest.mark.nightly
    def test_prelu_opset7_shared_channels(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, opset=7, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_prelu_shared_channels_precommit(self, params, ie_device, precision, ir_version,
                                             temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_scalar_precommit)
    @pytest.mark.precommit
    def test_prelu_scalar_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_scalar)
    @pytest.mark.nightly
    def test_prelu_scalar(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_shared_channels)
    @pytest.mark.nightly
    def test_prelu_shared_channels(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

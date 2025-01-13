# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestLeakyRelu(OnnxRuntimeLayerTest):
    def create_net(self, shape, alpha, ir_version):
        """
            ONNX net                         IR net

            Input->LeakyRelu->Output   =>    Input->ReLU

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
            'LeakyRelu',
            inputs=['input'],
            outputs=['output'],
            alpha=alpha
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #

        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'alpha_indata': {'value': alpha, 'kind': 'data'},
                'alpha': {'kind': 'op', 'type': 'Const'},
                'alpha_data': {'shape': [1], 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'PReLU'},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('alpha_indata', 'alpha'),
                                   ('alpha', 'alpha_data'),
                                   ('alpha_data', 'node'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')])

        return onnx_net, ref_net

    def create_net_const(self, shape, alpha, precision, ir_version):
        """
            ONNX net                                        IR net

            Input->Concat(+LeakyRelu const)->Output   =>    Input->Concat(+ReLU const)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto
        import numpy as np

        concat_axis = 0
        output_shape = shape.copy()
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        const_number = np.prod(shape)
        constant = np.random.randint(-127, 127, const_number).astype(float)
        constant = np.reshape(constant, shape)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=constant.shape,
                vals=constant.flatten(),
            ),
        )

        node_def = onnx.helper.make_node(
            'LeakyRelu',
            inputs=['const1'],
            outputs=['lrelu1'],
            alpha=alpha
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'lrelu1'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        constant = np.clip(constant, 0, np.inf) + np.clip(constant, -np.inf, 0) * alpha
        if precision == 'FP16':
            constant = constant.astype(np.float16)

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'const_indata': {'shape': None, 'kind': 'data'},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': shape, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('const_indata', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'concat'),
                                   ('const_data', 'concat'),
                                   ('concat', 'concat_data'),
                                   ('concat_data', 'result')])

        return onnx_net, ref_net

    test_data_precommit = [
        dict(shape=[1, 2], alpha=0.1),
        dict(shape=[2, 3, 4, 5, 6], alpha=4.5)]

    test_data = [
        dict(shape=[10, 12], alpha=0.1),
        dict(shape=[8, 10, 12], alpha=0.9),
        dict(shape=[6, 8, 10, 12], alpha=1.5),
        dict(shape=[4, 6, 8, 10, 12], alpha=4.5)]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_leaky_relu_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_leaky_relu(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_leaky_relu_const_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_leaky_relu_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

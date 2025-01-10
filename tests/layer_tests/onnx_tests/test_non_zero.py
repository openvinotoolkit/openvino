# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestNonZero(OnnxRuntimeLayerTest):
    def create_net(self, shape, ir_version):
        """
            ONNX net                    IR net

            Input->NonZero->Output   =>    Input->NonZero->Result

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
            'NonZero',
            inputs=['input'],
            outputs=['output']
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
                'node': {'kind': 'op', 'type': 'NonZero', 'version': 'opset3',
                         'output_type': 'i64'},
                'node_data': {'shape': [len(shape), np.prod(shape)], 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])
        return onnx_net, ref_net

    def create_net_const(self, input_value, output_value, precision, ir_version):
        """
            ONNX net                                   IR net

            Input->Concat(+NonZero const)->Output   =>    Input->Concat(+const)->Result

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        concat_axis = 0
        output_shape = list(output_value.shape)
        output_shape[concat_axis] *= 2

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, output_value.shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_const_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['const1'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=input_value.shape,
                vals=input_value.flatten(),
            ),
        )

        node_def = onnx.helper.make_node(
            'NonZero',
            inputs=['const1'],
            outputs=['nonzero1']
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'nonzero1'],
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
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': output_value.shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': output_value.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': output_value.shape, 'kind': 'data'},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_data', 'concat'),
                                   ('const_data', 'concat'),
                                   ('concat', 'concat_data'),
                                   ('concat_data', 'result')
                                   ])

        return onnx_net, ref_net

    test_data = [
        dict(shape=[10, 12]),
        dict(shape=[8, 10, 12]),
        dict(shape=[6, 8, 10, 12]),
        dict(shape=[4, 6, 8, 10, 12])
    ]

    test_const_data = [
        dict(
            input_value=np.array([3, 0, 0, 0, 4, 0, 5, 6, 0]).reshape((3, 3)),
            output_value=np.array([0, 1, 2, 2, 0, 1, 0, 1]).reshape(2, 4),
        ),
        dict(
            input_value=np.array([0, 1, 0, 1]).reshape((4)),
            output_value=np.array([1, 3]).reshape((1, 2)),
        ),
        dict(
            input_value=np.array([0, 1, 0, 1, 1, 0, 1, 0]).reshape((2, 4)),
            output_value=np.array([0, 0, 1, 1, 1, 3, 0, 2]).reshape((2, 4)),
        ),
        dict(
            input_value=np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0]).reshape(
                (2, 3, 3)),
            output_value=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2,
                                   0, 0, 0, 1, 1, 2, 0, 2, 1, 0, 1, 2, 0, 1, 2, 0, 2, 1]).reshape(
                (3, 12)),
        ),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_non_zero(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_const_data)
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_non_zero_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, precision=precision, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

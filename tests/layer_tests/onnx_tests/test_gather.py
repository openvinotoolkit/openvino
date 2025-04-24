# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestGather(OnnxRuntimeLayerTest):
    def create_net(self, shape, axis, indices, output_shape, ir_version):
        """
            ONNX net                   IR net

            Input->Gather->Output   =>    Input->Gather

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        indices = np.array(indices)

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        node_indices_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['indices'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.INT64,
                dims=indices.shape,
                vals=indices.flatten(),
            ),
        )

        args = dict()
        if axis:
            args['axis'] = axis
        else:
            axis = 0
        node_def = onnx.helper.make_node(
            'Gather',
            inputs=['input', 'indices'],
            outputs=['output'],
            **args
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_indices_def, node_def],
            'test_model',
            [input],
            [output]
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
                'input_const_data': {'kind': 'data', 'value': indices.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': indices.shape, 'kind': 'data', 'value': None},
                'input_axis_const_data': {'kind': 'data', 'value': [axis]},
                'axis_const': {'kind': 'op', 'type': 'Const'},
                'axis_const_data': {'shape': [], 'kind': 'data', 'value': None},
                'node': {'kind': 'op', 'type': 'Gather'},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_const_data', 'const'),
                                   ('const', 'const_data'),
                                   ('input_axis_const_data', 'axis_const'),
                                   ('axis_const', 'axis_const_data'),
                                   ('input_data', 'node'),
                                   ('const_data', 'node'),
                                   ('axis_const_data', 'node'),
                                   ('node', 'node_data'),
                                   ('node_data', 'result')
                                   ])

        return onnx_net, ref_net

    def create_net_const(self, shape, axis, indices, output_shape, ir_version):
        """
            ONNX net                                       IR net

            Input->Concat(+gathered const)->Output   =>    Input->Concat(+const)

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        indices = np.array(indices)

        concat_axis = 0
        input_shape = output_shape.copy()
        concat_output_shape = output_shape.copy()
        concat_output_shape[concat_axis] = 2 * concat_output_shape[concat_axis]
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, input_shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, concat_output_shape)

        constant = np.random.randint(-127, 127, shape).astype(float)

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

        node_indices_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['indices'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.INT64,
                dims=indices.shape,
                vals=indices.flatten(),
            ),
        )

        args = dict()
        if axis:
            args['axis'] = axis
        node_def = onnx.helper.make_node(
            'Gather',
            inputs=['const1', 'indices'],
            outputs=['gather'],
            **args
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'gather'],
            outputs=['output'],
            axis=concat_axis
        )

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            [node_const_def, node_indices_def, node_def, node_concat_def],
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')

        #
        #   Create reference IR net
        #
        constant = np.take(constant, indices, axis=axis if axis else 0)

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': constant.shape, 'kind': 'data', 'value': None},
                'concat': {'kind': 'op', 'type': 'Concat', 'axis': concat_axis},
                'concat_data': {'shape': concat_output_shape, 'kind': 'data'},
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

    test_data_precommit = [
        dict(shape=[6, 8, 10, 12], axis=2, indices=[[0, 2, 4], [5, 7, 9]],
             output_shape=[6, 8, 2, 3, 12]),
        dict(shape=[4, 6, 8, 10, 12], axis=1, indices=[2, 5], output_shape=[4, 2, 8, 10, 12]),
        dict(shape=[4, 6, 8, 10, 12], axis=-1, indices=[5, 8], output_shape=[4, 6, 8, 10, 2]),
        dict(shape=[6, 8, 10, 12], axis=-1, indices=[[[2, -1], [3, 2]], [[5, -1], [3, -2]]],
             output_shape=[6, 8, 10, 2, 2, 2])
    ]

    test_data = [dict(shape=[10, 12], axis=0, indices=[3, 6], output_shape=[2, 12]),
                 dict(shape=[10, 12], axis=-1, indices=[4, 7], output_shape=[10, 2]),
                 dict(shape=[10, 12], axis=None, indices=[[0, 1, 3, 4], [5, 6, 8, 9]],
                      output_shape=[2, 4, 12]),
                 dict(shape=[10, 12], axis=1, indices=[[0, 1, 3, 4, 5], [6, 7, 9, 10, 11]],
                      output_shape=[10, 2, 5]),
                 dict(shape=[8, 10, 12], axis=0, indices=[3, 6], output_shape=[2, 10, 12]),
                 dict(shape=[8, 10, 12], axis=-1, indices=[5, 8], output_shape=[8, 10, 2]),
                 dict(shape=[8, 10, 12], axis=None, indices=[[0, 1], [3, 4], [6, 7]],
                      output_shape=[3, 2, 10, 12]),
                 dict(shape=[8, 10, 12], axis=1, indices=[[0, 2, 4], [5, 7, 9]],
                      output_shape=[8, 2, 3, 12]),
                 dict(shape=[6, 8, 10, 12], axis=-1, indices=[5, 8], output_shape=[6, 8, 10, 2]),
                 dict(shape=[6, 8, 10, 12], axis=None, indices=[[0, 1, 2], [3, 4, 5]],
                      output_shape=[2, 3, 8, 10, 12]),
                 dict(shape=[6, 8, 10, 12], axis=2, indices=[[0, 2, 4], [5, 7, 9]],
                      output_shape=[6, 8, 2, 3, 12]),
                 dict(shape=[4, 6, 8, 10, 12], axis=0, indices=[1, 3],
                      output_shape=[2, 6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12], axis=1, indices=[2, 5],
                      output_shape=[4, 2, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12], axis=-1, indices=[5, 8],
                      output_shape=[4, 6, 8, 10, 2])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_gather(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_gather(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_gather_const(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net_const(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

    test_data_negative_indices = [
        dict(shape=[10, 12], axis=0, indices=[3, -1, -4], output_shape=[3, 12]),
        dict(shape=[6, 10, 14, 12], axis=1, indices=[[0, -1, 3, -4], [-5, 6, -7, 8]],
             output_shape=[6, 2, 4, 14, 12]),
        dict(shape=[8, 10, 14, 12], axis=1, indices=[[-2, 2, -4], [5, -7, 9]],
             output_shape=[8, 2, 3, 14, 12]),
        dict(shape=[6, 8, 10, 12], axis=-1, indices=[[[2, -1], [3, 2]], [[5, -1], [3, -2]]],
             output_shape=[6, 8, 10, 2, 2, 2])]

    @pytest.mark.parametrize("params", test_data_negative_indices)
    @pytest.mark.nightly
    def test_gather_nightly_negative_indices(self, params, ie_device, precision, ir_version,
                                             temp_dir):
        if ie_device == 'GPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

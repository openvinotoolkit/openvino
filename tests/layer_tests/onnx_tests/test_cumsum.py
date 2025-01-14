# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


def cumsum(a, axis=None, exclusive=False, reverse=False):
    if reverse:
        a = np.flip(a, axis)
    res = np.cumsum(a, axis=axis)
    if exclusive:
        res -= a
    if reverse:
        res = np.flip(res, axis)
    return res


class TestCumSum(OnnxRuntimeLayerTest):
    def create_net(self, shape, ir_version, axis=None, reverse=None, exclusive=None):
        """
            ONNX net                      IR net

            Input->CumSum->Output   =>    Input->CumSum

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, shape)

        nodes = []
        inputs = ['input']
        if axis is not None:
            node_axis_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['axis'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[],
                    vals=[axis],
                ),
            )
            nodes.append(node_axis_def)
            inputs.append('axis')

        args = dict()
        if exclusive is not None:
            args['exclusive'] = exclusive
        if reverse is not None:
            args['reverse'] = reverse
        node_def = onnx.helper.make_node(
            'CumSum',
            inputs=inputs,
            outputs=['output'],
            **args
        )
        nodes.append(node_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')
        onnx.checker.check_model(onnx_net)

        #
        #   Create reference IR net
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'node': {'kind': 'op', 'type': 'CumSum'},
                'node_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            if exclusive is not None:
                nodes_attributes['node']['exclusive'] = exclusive
            if reverse is not None:
                nodes_attributes['node']['reverse'] = reverse
            edges = [('input', 'input_data'),
                     ('input_data', 'node'),
                     ('node', 'node_data'),
                     ('node_data', 'result')
                     ]
            if axis is not None:
                nodes_attributes.update({
                    'input_axis_data': {'kind': 'data', 'value': [axis]},
                    'axis': {'kind': 'op', 'type': 'Const'},
                    'axis_data': {'shape': [], 'kind': 'data'}})
                edges.extend([('input_axis_data', 'axis'),
                              ('axis', 'axis_data'),
                              ('axis_data', 'node')])
            ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    def create_net_const(self, shape, precision, ir_version, axis=None, reverse=None,
                         exclusive=None):
        """
            ONNX net                                     IR net

            Input->Concat(+cumsum const)->Output   =>    Input->Concat(+const)

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

        constant = np.random.randn(*shape).astype(float)

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

        nodes = [node_const_def]
        inputs = ['const1']
        if axis is not None:
            node_axis_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['axis'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.INT64,
                    dims=[],
                    vals=[axis],
                ),
            )
            nodes.append(node_axis_def)
            inputs.append('axis')

        args = dict()
        if exclusive is not None:
            args['exclusive'] = exclusive
        if reverse is not None:
            args['reverse'] = reverse
        node_def = onnx.helper.make_node(
            'CumSum',
            inputs=inputs,
            outputs=['cumsum'],
            **args
        )

        node_concat_def = onnx.helper.make_node(
            'Concat',
            inputs=['input', 'cumsum'],
            outputs=['output'],
            axis=concat_axis
        )
        nodes.extend([node_def, node_concat_def])

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
            'test_model',
            [input],
            [output],
        )

        # Create the model (ModelProto)
        onnx_net = onnx_make_model(graph_def, producer_name='test_model')
        onnx.checker.check_model(onnx_net)

        #
        #   Create reference IR net
        #
        constant = cumsum(constant, axis=axis, reverse=reverse, exclusive=exclusive)
        ref_net = None
        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'input_const_data': {'kind': 'data', 'value': constant.flatten()},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': shape, 'kind': 'data'},
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
        dict(shape=[1]),
        dict(shape=[1, 2]),
        dict(shape=[2, 4, 6]),
        dict(shape=[2, 4, 6, 8]),
        dict(shape=[2, 4, 6, 8, 10]),
        dict(shape=[1, 2], axis=-2),
        dict(shape=[1, 2], axis=1),
        dict(shape=[2, 4, 6], axis=-3),
        dict(shape=[2, 4, 6], axis=2),
        dict(shape=[2, 4, 6, 8], axis=-4),
        dict(shape=[2, 4, 6, 8], axis=3),
        dict(shape=[2, 4, 6, 8, 10], axis=-1),
        dict(shape=[2, 4, 6, 8, 10], axis=4)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("reverse", [0, 1])
    @pytest.mark.parametrize("exclusive", [0, 1])
    @pytest.mark.nightly
    def test_cumsum(self, params, reverse, exclusive, ie_device, precision, ir_version, temp_dir):
        if 'axis' not in params:
            pytest.skip('No axis cases fail in ONNX')
        elif 'axis' in params and params['axis'] == -2 and exclusive == 1:
            pytest.skip('Disabled due to an exception thrown by ONNXRuntime for this use case')
        self._test(
            *self.create_net(**params, exclusive=exclusive, reverse=reverse, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("reverse", [0, 1])
    @pytest.mark.parametrize("exclusive", [0, 1])
    @pytest.mark.nightly
    def test_cumsum_const(self, params, reverse, exclusive, ie_device, precision, ir_version,
                          temp_dir):
        if 'axis' not in params:
            pytest.skip('No axis cases fail in ONNX')
        elif 'axis' in params and params['axis'] == -2 and exclusive == 1:
            pytest.skip('Disabled due to an exception thrown by ONNXRuntime for this use case')
        self._test(*self.create_net_const(**params, precision=precision, exclusive=exclusive,
                                          reverse=reverse,
                                          ir_version=ir_version), ie_device, precision, ir_version,
                   temp_dir=temp_dir)

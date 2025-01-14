# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestArgMax(OnnxRuntimeLayerTest):
    def create_net(self, shape, axis, keepdims, ir_version):
        """
            ONNX net                      IR net

            Input->ArgMax->Output   =>    Input->TopK

        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        output_shape = shape.copy()
        output_shape[axis if axis is not None else 0] = 1
        output_shape_squeeze = output_shape.copy()
        if keepdims == 0:
            output_shape_squeeze.remove(1)
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.INT64, output_shape_squeeze)

        const = np.random.randint(-10, 10, output_shape_squeeze).astype(np.int64)

        args = dict()
        if axis is not None:
            args['axis'] = axis
        else:
            axis = 0
        if keepdims is not None:
            args['keepdims'] = keepdims
        node_def = onnx.helper.make_node(
            'ArgMax',
            inputs=['input'],
            outputs=['argmax' if keepdims is None or keepdims == 1 else 'output'],
            **args
        )
        edges = [node_def]

        if keepdims is None or keepdims == 1:
            node_flatten_def = onnx.helper.make_node(
                'Flatten',
                inputs=['argmax'],
                outputs=['output']
            )
            edges.append(node_flatten_def)

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            edges,
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
                'const_indata': {'shape': [1], 'kind': 'data'},
                'const': {'kind': 'op', 'type': 'Const'},
                'const_data': {'shape': [], 'kind': 'data'},  # TODO shape [] or [1] ??
                'node': {'kind': 'op', 'type': 'TopK'},
                'node_data': {'shape': output_shape, 'kind': 'data'},
                'indices_data': {'shape': output_shape, 'kind': 'data'},
                'result1': {'kind': 'op', 'type': 'Result'},
                'result2': {'kind': 'op', 'type': 'Result'}
            }
            edges = [('input', 'input_data'),
                     ('const_indata', 'const'),
                     ('const', 'const_data'),
                     ('input_data', 'node'),
                     ('const_data', 'node'),
                     ('node', 'node_data'),
                     ('node', 'indices_data'),
                     ('node_data', 'result1')]
            if keepdims == 0:
                nodes_attributes.update({'squeeze_const_indata': {'shape': [1], 'kind': 'data'},
                                         'squeeze_const': {'kind': 'op', 'type': 'Const'},
                                         'squeeze_const_data': {'shape': [1], 'kind': 'data'},
                                         'squeeze': {'kind': 'op', 'type': 'Squeeze'},
                                         'squeeze_data': {'shape': output_shape_squeeze,
                                                          'kind': 'data'}
                                         })
                edges.extend([('squeeze_const_indata', 'squeeze_const'),
                              ('squeeze_const', 'squeeze_const_data'),
                              ('indices_data', 'squeeze'),
                              ('squeeze_const_data', 'squeeze'),
                              ('squeeze', 'squeeze_data'),
                              ('squeeze_data', 'result2')])
            else:
                nodes_attributes.update(
                    {'flatten_const_indata': {'kind': 'data', 'value': [0, -1]},
                     'flatten_const': {'kind': 'op', 'type': 'Const'},
                     'flatten_const_data': {'shape': [2], 'kind': 'data'},
                     'flatten': {'kind': 'op', 'type': 'Reshape'},
                     'flatten_data': {
                         'shape': [output_shape_squeeze[0], np.prod(output_shape_squeeze[1:])],
                         'kind': 'data'}
                     })
                edges.extend([('indices_data', 'flatten'),
                              ('flatten_const_indata', 'flatten_const'),
                              ('flatten_const', 'flatten_const_data'),
                              ('flatten_const_data', 'flatten'),
                              ('flatten', 'flatten_data'),
                              ('flatten_data', 'result2')])

            ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    test_data = [
        dict(shape=[10, 12], axis=None),
        dict(shape=[10, 12], axis=1),
        dict(shape=[8, 10, 12], axis=None),
        dict(shape=[8, 10, 12], axis=1),
        dict(shape=[8, 10, 12], axis=2),
        dict(shape=[6, 8, 10, 12], axis=None),
        dict(shape=[6, 8, 10, 12], axis=1),
        dict(shape=[6, 8, 10, 12], axis=2),
        dict(shape=[6, 8, 10, 12], axis=3),
        dict(shape=[4, 6, 8, 10, 12], axis=None),
        dict(shape=[4, 6, 8, 10, 12], axis=1),
        dict(shape=[4, 6, 8, 10, 12], axis=2),
        dict(shape=[4, 6, 8, 10, 12], axis=3),
        dict(shape=[4, 6, 8, 10, 12], axis=4)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keepdims", [None, 0])
    @pytest.mark.nightly
    def test_argmax(self, params, keepdims, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'CPU':
            pytest.skip('GREEN_SUITE')
        self._test(*self.create_net(**params, ir_version=ir_version, keepdims=keepdims),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

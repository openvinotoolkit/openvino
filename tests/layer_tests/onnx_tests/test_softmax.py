# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from unit_tests.utils.graph import build_graph


def second_input_data_of_reshape(src_shape, axis):
    if axis == 0:
        return [1, -1]
    if axis == 1:
        return [0, -1]
    if axis > 1:
        return [int(np.prod(int64_array(src_shape[: axis]))), -1]
    return [-1, int(np.prod(int64_array(src_shape[len(src_shape) + axis:])))]


def get_flatten_shape(src_shape, axis):
    flatten_axis = axis if axis >= 0 else len(src_shape) + axis
    if flatten_axis == 0:
        fst_dim = 1
        snd_dim = int(np.prod(int64_array(src_shape)))
    elif flatten_axis == 1:
        fst_dim = src_shape[0]
        snd_dim = int(np.prod(int64_array(src_shape[1:])))
    else:
        fst_dim = int(np.prod(int64_array(src_shape[: flatten_axis])))
        snd_dim = int(np.prod(int64_array(src_shape[flatten_axis:])))
    return [fst_dim, snd_dim]


class TestSoftmax(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shape, softmax_axis, ir_version):
        """
            ONNX net                       IR net

            Input->Softmax->Output   =>    Input->Reshape->SoftMax->Reshape

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
            'Softmax',
            inputs=['input'],
            outputs=['output'],
            axis=softmax_axis
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

        converted_shape = shape if len(shape) != 1 else shape[0]
        flatten_shape = get_flatten_shape(shape, softmax_axis)
        reshape_data_val = second_input_data_of_reshape(shape, softmax_axis)

        if check_ir_version(10, None, ir_version):
            if len(shape) == 2 and shape == flatten_shape:
                ref_nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter', 'shape': converted_shape},
                    'input_data': {'shape': shape, 'kind': 'data', 'value': None},
                    'flatten_shape_val': {'shape': int64_array(reshape_data_val).shape,
                                          'kind': 'data',
                                          'value': int64_array(reshape_data_val)},
                    'flatten_shape': {'type': 'Const', 'kind': 'op', 'shape': 2},
                    'flatten_shape_data': {'shape': int64_array([2]), 'kind': 'data',
                                           'value': None},
                    'reshape': {'kind': 'op', 'type': 'Reshape'},
                    'reshape_data': {'kind': 'data', 'shape': flatten_shape, 'value': None},
                    'softmax': {'type': 'SoftMax', 'kind': 'op', 'axis': 1},
                    'softmax_data': {'shape': flatten_shape, 'kind': 'data', 'value': None},
                    'result': {'kind': 'op', 'type': 'Result'},
                }

                ref_edges = [
                    ('input', 'input_data'),
                    ('flatten_shape_val', 'flatten_shape'),
                    ('flatten_shape', 'flatten_shape_data'),
                    ('flatten_shape_data', 'reshape', {'in': 1}),
                    ('input_data', 'reshape', {'in': 0}),
                    ('reshape', 'reshape_data'),
                    ('reshape_data', 'softmax'),
                    ('softmax', 'softmax_data'),
                    ('softmax_data', 'result'),
                ]
            else:
                ref_nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter', 'shape': converted_shape},
                    'input_data': {'shape': shape, 'kind': 'data', 'value': None},
                    'flatten_shape_val': {'shape': int64_array(reshape_data_val).shape,
                                          'kind': 'data',
                                          'value': int64_array(reshape_data_val)},
                    'flatten_shape': {'type': 'Const', 'kind': 'op', 'shape': 2},
                    'flatten_shape_data': {'shape': int64_array([2]), 'kind': 'data',
                                           'value': None},
                    'reshape': {'kind': 'op', 'type': 'Reshape'},
                    'reshape_data': {'kind': 'data', 'shape': flatten_shape, 'value': None},
                    'softmax': {'type': 'SoftMax', 'kind': 'op', 'axis': 1},
                    'softmax_data': {'shape': flatten_shape, 'kind': 'data', 'value': None},
                    'last_shape_val': {'shape': int64_array(shape).shape, 'kind': 'data',
                                       'value': int64_array(shape)},
                    'last_shape': {'type': 'Const', 'kind': 'op', 'shape': len(shape)},
                    'last_shape_data': {'shape': int64_array([len(shape)]), 'kind': 'data',
                                        'value': None},
                    'last_reshape': {'kind': 'op', 'type': 'Reshape'},
                    'last_reshape_data': {'kind': 'data', 'shape': shape, 'value': None},
                    'result': {'kind': 'op', 'type': 'Result'},
                }

                ref_edges = [
                    ('input', 'input_data'),
                    ('flatten_shape_val', 'flatten_shape'),
                    ('flatten_shape', 'flatten_shape_data'),
                    ('flatten_shape_data', 'reshape', {'in': 1}),
                    ('input_data', 'reshape', {'in': 0}),
                    ('reshape', 'reshape_data'),
                    ('reshape_data', 'softmax'),
                    ('softmax', 'softmax_data'),
                    ('last_shape_val', 'last_shape'),
                    ('last_shape', 'last_shape_data'),
                    ('last_shape_data', 'last_reshape', {'in': 1}),
                    ('softmax_data', 'last_reshape', {'in': 0}),
                    ('last_reshape', 'last_reshape_data'),
                    ('last_reshape_data', 'result'),
                ]

            ref_net = build_graph(ref_nodes_attributes, ref_edges)

        return onnx_net, ref_net

    test_data = [dict(shape=[10, 12], softmax_axis=0),
                 dict(shape=[10, 12], softmax_axis=1)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_softmax(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version,
                   temp_dir=temp_dir)

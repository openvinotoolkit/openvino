# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
pytest.importorskip("openvino.tools.mo", reason="Ticket - 157136")

from common.layer_test_class import check_ir_version
from common.onnx_layer_test_class import OnnxRuntimeLayerTest, onnx_make_model

from unit_tests.utils.graph import build_graph


class TestConv(OnnxRuntimeLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randn(*inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_net(self, shape, weights_shape, dilations, group, pads, strides, bias, ir_version,
                   auto_pad=None):
        """
            ONNX net                    IR net

            Input->Conv->Output   =>    Input->Convolution
        """

        #
        #   Create ONNX model
        #

        import onnx
        from onnx import helper
        from onnx import TensorProto

        output_shape = np.array(shape)
        output_shape[1] = group
        _pads = np.array(pads).reshape([2, -1])
        kernel_extent = np.array(dilations) * (np.array(weights_shape[2:]) - 1) + 1
        spatial_val_wo_stride = shape[2:] + np.add(_pads[0, :], _pads[1, :]) - kernel_extent
        output_shape[2:] = (spatial_val_wo_stride.astype(float) / strides + 1).astype(np.int64)
        output_shape = output_shape.astype(int).tolist()
        input = helper.make_tensor_value_info('input', TensorProto.FLOAT, shape)
        output = helper.make_tensor_value_info('output', TensorProto.FLOAT, output_shape)

        weights_const = np.random.randn(*weights_shape).astype(np.float32)

        node_weights_def = onnx.helper.make_node(
            'Constant',
            inputs=[],
            outputs=['weights'],
            value=helper.make_tensor(
                name='const_tensor',
                data_type=TensorProto.FLOAT,
                dims=weights_const.shape,
                vals=weights_const.flatten(),
            ),
        )

        conv_args = dict(kernel_shape=weights_shape[2:],
                         dilations=dilations,
                         group=group,
                         strides=strides)
        if pads and auto_pad not in ['SAME_UPPER', 'SAME_LOWER']:
            conv_args['pads'] = pads
        if auto_pad:
            conv_args['auto_pad'] = auto_pad
        if bias:
            bias_const = np.random.randint(-10, 10, weights_shape[0]).astype(np.float32)

            node_bias_def = onnx.helper.make_node(
                'Constant',
                inputs=[],
                outputs=['bias'],
                value=helper.make_tensor(
                    name='const_tensor',
                    data_type=TensorProto.FLOAT,
                    dims=bias_const.shape,
                    vals=bias_const.flatten(),
                ),
            )
            node_def = onnx.helper.make_node(
                'Conv',
                inputs=['input', 'weights', 'bias'],
                outputs=['output'],
                **conv_args
            )
            nodes = [node_weights_def, node_bias_def, node_def]
        else:
            node_def = onnx.helper.make_node(
                'Conv',
                inputs=['input', 'weights'],
                outputs=['output'],
                **conv_args
            )
            nodes = [node_weights_def, node_def]

        # Create the graph (GraphProto)
        graph_def = helper.make_graph(
            nodes,
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
            if len(shape) == 3:
                input_shape = shape.copy()
                input_shape.insert(2, 1)
                node_shape = output_shape.copy()
                node_shape.insert(2, 1)
                nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter'},
                    'input_data': {'shape': shape, 'kind': 'data'},
                    'before_shape_const_indata': {'shape': [len(input_shape)], 'value': input_shape,
                                                  'kind': 'data'},
                    'before_shape_const': {'kind': 'op', 'type': 'Const'},
                    'before_shape_const_data': {'shape': [len(input_shape)], 'kind': 'data'},
                    'reshape_before': {'kind': 'op', 'type': 'Reshape'},
                    'reshape_before_data': {'shape': input_shape, 'kind': 'data'},
                    'kernel_indata': {'kind': 'data', 'shape': [len(weights_const.flatten())]},
                    'kernel': {'kind': 'op', 'type': 'Const'},
                    'kernel_data': {'kind': 'data', 'value': None},
                    'node': {'kind': 'op',
                             'type': 'Convolution' if group == 1 else 'GroupConvolution',
                             'dilations': [1, dilations[0]],
                             'pads_begin': [0, _pads[0, 0]], 'pads_end': [0, _pads[1, 0]]},
                    'node_data': {'shape': node_shape, 'kind': 'data'},
                    'after_shape_const_indata': {'shape': [len(output_shape)],
                                                 'value': output_shape, 'kind': 'data'},
                    'after_shape_const': {'kind': 'op', 'type': 'Const'},
                    'after_shape_const_data': {'shape': [len(output_shape)], 'kind': 'data'},
                    'reshape_after': {'kind': 'op', 'type': 'Reshape'},
                    'reshape_after_data': {'shape': output_shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}}
                edges = [('input', 'input_data'),
                         ('input_data', 'reshape_before'),
                         ('before_shape_const_indata', 'before_shape_const'),
                         ('before_shape_const', 'before_shape_const_data'),
                         ('before_shape_const_data', 'reshape_before'),
                         ('reshape_before', 'reshape_before_data'),
                         ('reshape_before_data', 'node'),
                         ('kernel_indata', 'kernel'),
                         ('kernel', 'kernel_data'),
                         ('kernel_data', 'node'),
                         ('node', 'node_data'),
                         ('node_data', 'reshape_after'),
                         ('after_shape_const_indata', 'after_shape_const'),
                         ('after_shape_const', 'after_shape_const_data'),
                         ('after_shape_const_data', 'reshape_after'),
                         ('reshape_after', 'reshape_after_data')]
                if bias:
                    nodes_attributes.update(
                        {'const_indata': {'kind': 'data', 'value': bias_const.flatten()},
                         'const': {'kind': 'op', 'type': 'Const'},
                         'const_data': {'kind': 'data', 'shape': None},
                         'bias': {'type': 'Add', 'kind': 'op'},
                         'bias_data': {'kind': 'data', 'shape': output_shape}})
                    edges += [('reshape_after_data', 'bias'),
                              ('const_indata', 'const'),
                              ('const', 'const_data'),
                              ('const_data', 'bias'),
                              ('bias', 'bias_data'),
                              ('bias_data', 'result')]
                else:
                    edges += [('reshape_after_data', 'result')]
                ref_net = build_graph(nodes_attributes, edges)
            else:
                _weights_shape = weights_shape.copy()
                if group != 1:
                    _weights_shape.insert(1, 1)
                nodes_attributes = {
                    'input': {'kind': 'op', 'type': 'Parameter'},
                    'input_data': {'shape': shape, 'kind': 'data'},
                    'kernel_indata': {'kind': 'data', 'value': weights_const.flatten()},
                    'kernel': {'kind': 'op', 'type': 'Const'},
                    'kernel_data': {'kind': 'data', 'shape': _weights_shape},
                    'node': {'kind': 'op',
                             'type': 'Convolution' if group == 1 else 'GroupConvolution',
                             'dilations': dilations, 'pads_begin': _pads[0, :],
                             'pads_end': _pads[1, :]},
                    'node_data': {'shape': output_shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}}
                edges = [('input', 'input_data'),
                         ('input_data', 'node'),
                         ('kernel_indata', 'kernel'),
                         ('kernel', 'kernel_data'),
                         ('kernel_data', 'node'),
                         ('node', 'node_data')]

                if bias:
                    nodes_attributes.update(
                        {'const_indata': {'kind': 'data', 'value': bias_const.flatten()},
                         'const': {'kind': 'op', 'type': 'Const'},
                         'const_data': {'kind': 'data', 'shape': None},
                         'bias': {'type': 'Add', 'kind': 'op'},
                         'bias_data': {'kind': 'data', 'shape': output_shape}})
                    edges += [('node_data', 'bias'),
                              ('const_indata', 'const'),
                              ('const', 'const_data'),
                              ('const_data', 'bias'),
                              ('bias', 'bias_data'),
                              ('bias_data', 'result')]
                else:
                    edges += [('node_data', 'result')]

                ref_net = build_graph(nodes_attributes, edges)

        return onnx_net, ref_net

    test_data_3D = [
        dict(weights_shape=[1, 3, 3], group=1),
        dict(weights_shape=[1, 3, 5], group=1),
        dict(weights_shape=[3, 1, 3], group=3),
        dict(weights_shape=[3, 1, 5], group=3)]

    test_data_3D_autopad = [
        dict(weights_shape=[1, 3, 3], group=1, pads=[1, 1], strides=[1], dilations=[1]),
        dict(weights_shape=[1, 3, 3], group=1, pads=[2, 2], strides=[1], dilations=[2]),
        dict(weights_shape=[1, 3, 3], group=1, pads=[1, 1], strides=[2], dilations=[1]),
        dict(weights_shape=[1, 3, 3], group=1, pads=[2, 2], strides=[2], dilations=[2]),
        dict(weights_shape=[1, 3, 5], group=1, pads=[2, 2], strides=[1], dilations=[1]),
        dict(weights_shape=[1, 3, 5], group=1, pads=[4, 4], strides=[1], dilations=[2]),
        dict(weights_shape=[1, 3, 5], group=1, pads=[2, 2], strides=[2], dilations=[1]),
        dict(weights_shape=[1, 3, 5], group=1, pads=[4, 4], strides=[2], dilations=[2]),
        dict(weights_shape=[3, 1, 3], group=3, pads=[1, 1], strides=[1], dilations=[1]),
        dict(weights_shape=[3, 1, 3], group=3, pads=[2, 2], strides=[1], dilations=[2]),
        dict(weights_shape=[3, 1, 3], group=3, pads=[1, 1], strides=[2], dilations=[1]),
        dict(weights_shape=[3, 1, 3], group=3, pads=[2, 2], strides=[2], dilations=[2]),
        dict(weights_shape=[3, 1, 5], group=3, pads=[2, 2], strides=[1], dilations=[1]),
        dict(weights_shape=[3, 1, 5], group=3, pads=[4, 4], strides=[1], dilations=[2]),
        dict(weights_shape=[3, 1, 5], group=3, pads=[2, 2], strides=[2], dilations=[1]),
        dict(weights_shape=[3, 1, 5], group=3, pads=[4, 4], strides=[2], dilations=[2])]

    test_data_4D_precommit = [
        dict(weights_shape=[1, 3, 3, 3], group=1),
        dict(weights_shape=[3, 1, 3, 3], group=3)]

    test_data_4D = [
        dict(weights_shape=[1, 3, 3, 3], group=1),
        dict(weights_shape=[1, 3, 5, 3], group=1),
        dict(weights_shape=[3, 1, 3, 3], group=3),
        dict(weights_shape=[3, 1, 3, 5], group=3)]

    test_data_4D_autopad = [
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[1, 1, 1, 1], strides=[1, 1],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[2, 2, 2, 2], strides=[1, 1],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[3, 5, 3, 5], strides=[1, 1],
             dilations=[3, 5]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[1, 1, 1, 1], strides=[2, 2],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[2, 2, 2, 2], strides=[2, 2],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[3, 5, 3, 5], strides=[2, 2],
             dilations=[3, 5]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[1, 0, 1, 0], strides=[3, 5],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[2, 0, 2, 0], strides=[3, 5],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 3, 3], group=1, pads=[3, 3, 3, 3], strides=[3, 5],
             dilations=[3, 5]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[2, 1, 2, 1], strides=[1, 1],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[4, 2, 4, 2], strides=[1, 1],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[6, 5, 6, 5], strides=[1, 1],
             dilations=[3, 5]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[2, 1, 2, 1], strides=[2, 2],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[4, 2, 4, 2], strides=[2, 2],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[6, 5, 6, 5], strides=[2, 2],
             dilations=[3, 5]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[2, 0, 2, 0], strides=[3, 5],
             dilations=[1, 1]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[4, 0, 4, 0], strides=[3, 5],
             dilations=[2, 2]),
        dict(weights_shape=[1, 3, 5, 3], group=1, pads=[6, 3, 6, 3], strides=[3, 5],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[1, 1, 1, 1], strides=[1, 1],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[2, 2, 2, 2], strides=[1, 1],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[3, 5, 3, 5], strides=[1, 1],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[1, 1, 1, 1], strides=[2, 2],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[2, 2, 2, 2], strides=[2, 2],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[3, 5, 3, 5], strides=[2, 2],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[1, 0, 1, 0], strides=[3, 5],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[2, 0, 2, 0], strides=[3, 5],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 3], group=3, pads=[3, 3, 3, 3], strides=[3, 5],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[1, 2, 1, 2], strides=[1, 1],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[2, 4, 2, 4], strides=[1, 1],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[3, 10, 3, 10], strides=[1, 1],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[1, 2, 1, 2], strides=[2, 2],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[2, 4, 2, 4], strides=[2, 2],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[3, 10, 3, 10], strides=[2, 2],
             dilations=[3, 5]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[1, 0, 1, 0], strides=[3, 5],
             dilations=[1, 1]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[2, 2, 2, 2], strides=[3, 5],
             dilations=[2, 2]),
        dict(weights_shape=[3, 1, 3, 5], group=3, pads=[3, 8, 3, 8], strides=[3, 5],
             dilations=[3, 5])]

    test_data_5D_precommit = [
        dict(weights_shape=[1, 3, 3, 3, 3], group=1),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3)]

    test_data_5D = [
        dict(weights_shape=[1, 3, 3, 3, 3], group=1),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3)]

    test_data_5D_autopad = [
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[2, 2, 2, 2, 2, 2], strides=[1, 1, 1],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[3, 4, 5, 3, 4, 5], strides=[1, 1, 1],
             dilations=[3, 4, 5]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[1, 1, 1, 1, 1, 1], strides=[2, 2, 2],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[2, 2, 2, 2, 2, 2], strides=[2, 2, 2],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[3, 4, 5, 3, 4, 5], strides=[2, 2, 2],
             dilations=[3, 4, 5]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[1, 1, 0, 1, 1, 0], strides=[3, 4, 5],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[2, 2, 0, 2, 2, 0], strides=[3, 4, 5],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 3, 3], group=1, pads=[3, 4, 3, 3, 4, 3], strides=[3, 4, 5],
             dilations=[3, 4, 5]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[1, 1, 2, 1, 2, 2], strides=[1, 1, 1],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[2, 3, 4, 2, 3, 4], strides=[1, 1, 1],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[3, 6, 10, 3, 6, 10], strides=[1, 1, 1],
             dilations=[3, 4, 5]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[1, 1, 2, 1, 2, 2], strides=[2, 2, 2],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[2, 3, 4, 2, 3, 4], strides=[2, 2, 2],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[3, 6, 10, 3, 6, 10], strides=[2, 2, 2],
             dilations=[3, 4, 5]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[1, 1, 0, 1, 2, 0], strides=[3, 4, 5],
             dilations=[1, 1, 1]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[2, 3, 2, 2, 3, 2], strides=[3, 4, 5],
             dilations=[2, 2, 2]),
        dict(weights_shape=[1, 3, 3, 4, 5], group=1, pads=[3, 6, 8, 3, 6, 8], strides=[3, 4, 5],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[1, 1, 1, 1, 1, 1], strides=[1, 1, 1],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[2, 2, 2, 2, 2, 2], strides=[1, 1, 1],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[3, 4, 5, 3, 4, 5], strides=[1, 1, 1],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[1, 1, 1, 1, 1, 1], strides=[2, 2, 2],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[2, 2, 2, 2, 2, 2], strides=[2, 2, 2],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[3, 4, 5, 3, 4, 5], strides=[2, 2, 2],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[1, 1, 0, 1, 1, 0], strides=[3, 4, 5],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[2, 2, 0, 2, 2, 0], strides=[3, 4, 5],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 3, 3, 3], group=3, pads=[3, 4, 3, 3, 4, 3], strides=[3, 4, 5],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[2, 1, 1, 2, 2, 1], strides=[1, 1, 1],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[4, 3, 2, 4, 3, 2], strides=[1, 1, 1],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[6, 6, 5, 6, 6, 5], strides=[1, 1, 1],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[2, 1, 1, 2, 2, 1], strides=[2, 2, 2],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[4, 3, 2, 4, 3, 2], strides=[2, 2, 2],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[6, 6, 5, 6, 6, 5], strides=[2, 2, 2],
             dilations=[3, 4, 5]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[2, 1, 0, 2, 2, 0], strides=[3, 4, 5],
             dilations=[1, 1, 1]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[4, 3, 0, 4, 3, 0], strides=[3, 4, 5],
             dilations=[2, 2, 2]),
        dict(weights_shape=[3, 1, 5, 4, 3], group=3, pads=[6, 6, 3, 6, 6, 3], strides=[3, 4, 5],
             dilations=[3, 4, 5])]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.parametrize("dilations", [[1], [2]])
    @pytest.mark.parametrize("pads", [[0, 0], [1, 1], [1, 2]])
    @pytest.mark.parametrize("strides", [[1], [2]])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    def test_conv_3D(self, params, dilations, pads, strides, bias, ie_device, precision, ir_version,
                     temp_dir):
        self._test(*self.create_net(**params, shape=[2, 3, 25], dilations=dilations, pads=pads,
                                    strides=strides,
                                    bias=bias, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D_autopad[:-1])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER'])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_3D_autopad(self, params, auto_pad, bias, ie_device, precision, ir_version,
                             temp_dir):
        self._test(*self.create_net(**params, shape=[2, 3, 25], bias=bias, auto_pad=auto_pad,
                                    ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_precommit)
    @pytest.mark.parametrize("dilations", [[3, 5]])
    @pytest.mark.parametrize("pads", [[1, 2, 3, 4]])
    @pytest.mark.parametrize("strides", [[3, 5]])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.precommit
    def test_conv_4D_precommit(self, params, dilations, pads, strides, bias, ie_device, precision,
                               ir_version, temp_dir):
        self._test(*self.create_net(**params, shape=[2, 3, 25, 25], dilations=dilations, pads=pads,
                                    strides=strides,
                                    bias=bias, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.parametrize("dilations", [[1, 1], [2, 2], [3, 5]])
    @pytest.mark.parametrize("pads", [[0, 0, 0, 0], [1, 1, 1, 1], [1, 2, 3, 4]])
    @pytest.mark.parametrize("strides", [[1, 1], [2, 2], [3, 5]])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    def test_conv_4D(self, params, dilations, pads, strides, bias, ie_device, precision, ir_version,
                     temp_dir):
        self._test(
            *self.create_net(**params, shape=[2, 3, 25, 25], dilations=dilations, pads=pads,
                             strides=strides, bias=bias,
                             ir_version=ir_version), ie_device, precision, ir_version,
            temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D_autopad[:-1])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER'])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_4D_autopad(self, params, auto_pad, bias, ie_device, precision, ir_version,
                             temp_dir):
        self._test(*self.create_net(**params, shape=[2, 3, 25, 25], bias=bias, auto_pad=auto_pad,
                                    ir_version=ir_version), ie_device, precision, ir_version,
                   temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D_precommit)
    @pytest.mark.parametrize("dilations", [[3, 4, 5]])
    @pytest.mark.parametrize("pads", [[1, 2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("strides", [[3, 4, 5]])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.precommit
    def test_conv_5D_precommit(self, params, dilations, pads, strides, bias, ie_device, precision,
                               ir_version, temp_dir):
        custom_eps_value = 1e-1 if ie_device == 'GPU' and precision == 'FP16' else None
        self._test(
            *self.create_net(**params, shape=[2, 3, 25, 25, 25], dilations=dilations, pads=pads,
                             strides=strides,
                             bias=bias, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir, custom_eps=custom_eps_value)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.parametrize("dilations", [[1, 1, 1], [2, 2, 2], [3, 4, 5]])
    @pytest.mark.parametrize("pads", [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6]])
    @pytest.mark.parametrize("strides", [[1, 1, 1], [2, 2, 2], [3, 4, 5]])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_5D(self, params, dilations, pads, strides, bias, ie_device, precision, ir_version,
                     temp_dir):
        self._test(
            *self.create_net(**params, shape=[2, 3, 25, 25, 25], dilations=dilations, pads=pads,
                             strides=strides,
                             bias=bias, ir_version=ir_version),
            ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D_autopad[:-1])
    @pytest.mark.parametrize("auto_pad", ['SAME_UPPER', 'SAME_LOWER'])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.nightly
    @pytest.mark.skip(reason='GREEN_SUITE')
    def test_conv_5D_autopad(self, params, auto_pad, bias, ie_device, precision, ir_version,
                             temp_dir):
        self._test(
            *self.create_net(**params, shape=[2, 3, 25, 25, 25], bias=bias, auto_pad=auto_pad,
                             ir_version=ir_version), ie_device, precision, ir_version,
            temp_dir=temp_dir)

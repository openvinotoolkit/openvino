# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.back.ReshapeMutation import ReshapeMutation
from extensions.back.ReverseInputChannels import ApplyReverseChannels
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input, create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.strided_slice import StridedSlice
from mo.utils.error import Error


def resolve_convolution_with_group(node: Node, group: int, ir_version: str):
    input_shape = node.in_port(0).data.get_shape()
    assert len(input_shape) in [3, 4, 5]

    weights_shape = node.in_port(1).data.get_shape()
    assert weights_shape is not None
    assert len(weights_shape) in [3, 4, 5]
    assert weights_shape[0] % group == 0

    assert int64_array(node.output).ndim == 0
    if ir_version == 'V7':
        if weights_shape[0] == node.output:
            # weights are already is in [G*O I X Y] format
            return
        new_shape = int64_array([node.output, -1, *weights_shape[2:]])

    elif ir_version == 'V10':
        I = input_shape[1]
        new_shape = int64_array([group, node.output / group, I / group, *weights_shape[2:]])
        assert np.prod(weights_shape) == np.prod(new_shape), \
            'Initial weights shape {}, grouped weights shape {}'.format(weights_shape, new_shape)
        del node['group']
        node['type'] = 'GroupConvolution'
    else:
        raise Error("Unknown IR version: {}".format(ir_version))

    reshape = create_op_node_with_second_input(node.graph, Reshape, int64_array(new_shape),
                                               {'override_output_shape': True})

    node.in_port(1).get_connection().insert_node(reshape)


class ConvolutionNormalizer(BackReplacementPattern):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('node', dict(kind='op', type='Convolution'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['node']
        if node.has_valid('kernel_spatial'):
            del node['kernel_spatial']


class V7ConvolutionWithGroupsResolver(BackReplacementPattern):
    """
    Normalizes grouped convolution weights shape to fit special weights format [G*O I X Y]
    """
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Convolution'):
            group = node.soft_get('group', None)
            if group is not None:
                if group != 1 or node.soft_get('op') == 'DepthwiseConv2dNative':
                    resolve_convolution_with_group(node, group, ir_version='V7')


class V10ConvolutionWithGroupsResolver(BackReplacementPattern):
    """
    Normalizes grouped convolution weights shape to fit special weights format
        V10 IR:                 [G O I X Y]
    """
    enabled = False

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Convolution'):
            group = node.soft_get('group', None)
            if group is not None:
                if group != 1 or node.soft_get('op') == 'DepthwiseConv2dNative':
                    resolve_convolution_with_group(node, group, ir_version='V10')


class ConvolutionWithGroupsResolver(BackReplacementPattern):
    """
    Normalizes grouped convolution weights shape to fit special weights format
        V10 IR:                 [G O I X Y]
        lower IR versions:      [G*O I X Y]
    """
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation]

    def run_after(self):
        return [ApplyReverseChannels]

    def find_and_replace_pattern(self, graph: Graph):
        V7ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)
        PullReshapeThroughFQ().find_and_replace_pattern(graph)
        V10ConvolutionWithGroupsResolver().find_and_replace_pattern(graph)


class PullReshapeThroughFQ(BackReplacementPattern):
    """
    Before:
        ... -> FQ -> Reshape -> Convolution -> ...

    After:
        ... -> Reshape -> FQ (with aligned limits) -> Convolution -> ...
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('FQ', dict(type='FakeQuantize')),
                ('FQed', dict()),
                ('reshape', dict(type='Reshape')),
                ('reshaped', dict()),
                ('node', dict(type=lambda t: t in ['Convolution', 'GroupConvolution'])),
            ],
            edges=[
                ('FQ', 'FQed'),
                ('FQed', 'reshape', {'in': 0}),
                ('reshape', 'reshaped'),
                ('reshaped', 'node', {'in': 1}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        FQ = match['FQ']
        reshape = match['reshape']
        conv = match['node']

        rank_reshape = reshape.in_port(0).data.get_shape().size != reshape.out_port(0).data.get_shape().size

        if not all([np.prod(FQ.in_port(i).data.get_shape()) == 1 for i in range(1, 5)]):
            #  FakeQuantize has limits with multiple values, that should be reshaped too
            #  Pulling Reshape through such FQ is a complex procedure because of broadcasting rules
            return

        new_rank = reshape.out_port(0).data.get_shape().size

        reshape.in_port(0).disconnect()
        reshape.out_port(0).disconnect()

        FQ.out_port(0).connect(conv.in_port(1))
        FQ.in_port(0).get_connection().insert_node(reshape)

        reshape['need_shape_inference'] = True
        reshape['override_output_shape'] = True
        FQ['need_shape_inference'] = True
        FQ['override_output_shape'] = True

        if rank_reshape:
            # force rank of limit inputs to match 0-input rank
            # reshaping to lower range needs it the most due to FQ inner broadcast semantics
            for i in range(1, 5):
                reshape = create_op_node_with_second_input(graph, Reshape, int64_array([1] * new_rank),
                                                           {'override_output_shape': True})
                FQ.in_port(i).get_connection().insert_node(reshape)


class DeconvolutionNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation]

    def run_after(self):
        return [ApplyReverseChannels]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('node', dict(type='Deconvolution'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['node']
        node_name = node.soft_get('name', node.id)

        if 2 in node.in_ports() and not node.in_port(2).disconnected():
            # Third input represents output shape. Cutting its value according to scheme:
            # [N, C, spatial_dim_0, ..., spatial_dim_n] -> [spatial_dim_0, ..., spatial_dim_n]
            in_rank = node.in_port(0).data.get_shape().size

            shape_src = node.in_port(2).get_source()
            node.in_port(2).disconnect()

            ss_0 = create_op_with_const_inputs(graph, StridedSlice, {1: np.array([2], dtype=np.int32),
                                                                     2: np.array([in_rank], dtype=np.int32),
                                                                     3: np.array([1], dtype=np.int32)},
                                               {'name': node_name + '/ss_0_port',
                                                'begin_mask': np.array([1], dtype=np.int32),
                                                'end_mask': np.array([0], dtype=np.int32),
                                                'new_axis_mask': np.array([0], dtype=np.int32),
                                                'shrink_axis_mask': np.array([0], dtype=np.int32),
                                                'ellipsis_mask': np.array([0], dtype=np.int32)})

            shape_src.connect(ss_0.in_port(0))
            ss_0.out_port(0).connect(node.in_port(2))

            # Specification: *padding amount* is deduced from relation of input and output spatial shapes
            del node['pad']

        elif node.has_valid('original_output_spatial_shape'):
            # node had fixed output spatial shape set in original framework, so we restore it here
            const = Const(graph, {'value': int64_array(node.original_output_spatial_shape),
                                  'name': node_name + '/original_spatial_shape'}).create_node()
            node.add_input_port(2, skip_if_exist=True)
            const.out_port(0).connect(node.in_port(2))

            # Specification: *padding amount* is deduced from relation of input and output spatial shapes
            del node['pad']

        group = node.soft_get('group', 1)

        if group != 1:
            assert group > 1

            weights_shape = node.in_port(1).data.get_shape()
            assert weights_shape is not None
            I = node.in_port(0).data.get_shape()[1]
            assert I % group == 0
            assert node.output % group == 0

            new_shape = int64_array([group, I / group, node.output / group, *weights_shape[2:]])

            assert np.prod(weights_shape) == np.prod(new_shape), \
                'Initial weights shape {}, grouped weights shape {}'.format(weights_shape, new_shape)
            reshape = create_op_node_with_second_input(graph, Reshape, int64_array(new_shape),
                                                       {'override_output_shape': True},
                                                       node.in_port(1).get_source().node)

            node.in_port(1).get_connection().set_source(reshape.out_port(0))

            node['type'] = 'GroupConvolutionBackpropData'
        else:
            node['type'] = 'ConvolutionBackpropData'

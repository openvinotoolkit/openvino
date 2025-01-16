# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
import pytest

from openvino.tools.mo.back.ShuffleChannelPatternOptimization import ShuffleChannelFusion, DepthToSpaceFusion
from openvino.tools.mo.ops.depth_to_space import DepthToSpaceOp
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.ops.shufflechannel import ShuffleChannels
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.ops.reshape import Reshape
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, result, regular_op_with_shaped_data, \
    valued_const_with_data, connect, regular_op_with_empty_data


class TestShuffleChannelFusionTest():
    @staticmethod
    def get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, group):
        nodes = {
            **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter', 'shape': int64_array(input_shape),
                                                                 'infer': Parameter.infer}),

            **valued_const_with_data('reshape_0_pattern', int64_array(reshape_0_pattern)),
            **regular_op_with_empty_data('reshape_0', {'type': 'Reshape', 'infer': Reshape.infer}),

            **valued_const_with_data('order', int64_array(order)),
            **regular_op_with_empty_data('transpose', {'type': 'Transpose', 'infer': Transpose.infer}),

            **valued_const_with_data('reshape_1_pattern', int64_array(reshape_1_pattern)),
            **regular_op_with_empty_data('reshape_1', {'type': 'Reshape', 'infer': Reshape.infer,
                                                       'name': 'final_reshape'}),

            **result(),
        }
        edges = [
            *connect('input', '0:reshape_0'),
            *connect('reshape_0_pattern', '1:reshape_0'),
            *connect('reshape_0', '0:transpose'),
            *connect('order', '1:transpose'),
            *connect('transpose', '0:reshape_1'),
            *connect('reshape_1_pattern', '1:reshape_1'),
            *connect('reshape_1', 'output'),
        ]
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        for node in graph.get_op_nodes():
            node['op'] = node['type']
        graph.clean_up()

        ref_nodes = {
            **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter', 'shape': int64_array(input_shape),
                                                                 'infer': Parameter.infer}),
            **regular_op_with_empty_data('shuffle_channel', {'type': 'ShuffleChannels', 'infer': ShuffleChannels.infer,
                                                             'name': 'final_reshape', 'group': group}),
            **result()
        }
        ref_edges = [*connect('input', 'shuffle_channel'), *connect('shuffle_channel', 'output')]
        graph_ref = build_graph(ref_nodes, ref_edges, nodes_with_edges_only=True)
        for node in graph_ref.get_op_nodes():
            node['op'] = node['type']
        graph_ref.clean_up()

        return graph, graph_ref

    @pytest.mark.parametrize("input_shape, reshape_0_pattern, order, reshape_1_pattern, group",[
        ([1, 512, 7, 6], [1, 2, 256, 7, 6], [0, 2, 1, 3, 4], [1, 512, 7, 6], 2),
        ([2, 512, 7, 6], [2, 2, 256, 7, 6], [0, 2, 1, 3, 4], [2, 512, 7, 6], 2),
        ([1, 200, 200, 200], [1, 50, 4, 200, 200], [0, 2, 1, 3, 4], [1, 200, 200, 200], 50),
    ])
    def test_fusion(self, input_shape, reshape_0_pattern, order, reshape_1_pattern, group):
        graph, graph_ref = self.get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, group)
        ShuffleChannelFusion().find_and_replace_pattern(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        assert flag, resp
        assert len(graph.get_op_nodes(name='final_reshape')) == 1 and \
                        graph.get_op_nodes(name='final_reshape')[0].op == 'ShuffleChannels'

    @pytest.mark.parametrize("input_shape, reshape_0_pattern, order, reshape_1_pattern, group",[
        ([1, 512, 7, 6], [0, 2, 256, 7, 6], [0, 2, 1, 3, 4], [1, 512, 7, 6], 2),
        ([1, 512, 7, 6], [1, 2, 256, 7, 6], [0, 2, 1, 4, 3], [1, 512, 7, 6], 2),
        ([1, 512, 7, 6], [1, 2, 256, 7, 6], [0, 2, 1, 3, 4], [-1, 512, 7, 6], 2),
    ])
    def test_negative(self, input_shape, reshape_0_pattern, order, reshape_1_pattern, group):
        graph, _ = self.get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, group)
        graph_ref = graph.copy()
        ShuffleChannelFusion().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        assert flag, resp


class TestDepthToSpaceFusionTest():
    @staticmethod
    def get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, block_size):
        nodes = {
            **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter', 'shape': int64_array(input_shape),
                                                                 'infer': Parameter.infer}),

            **valued_const_with_data('reshape_0_pattern', int64_array(reshape_0_pattern)),
            **regular_op_with_empty_data('reshape_0', {'type': 'Reshape', 'infer': Reshape.infer}),

            **valued_const_with_data('order', int64_array(order)),
            **regular_op_with_empty_data('transpose', {'type': 'Transpose', 'infer': Transpose.infer}),

            **valued_const_with_data('reshape_1_pattern', int64_array(reshape_1_pattern)),
            **regular_op_with_empty_data('reshape_1', {'type': 'Reshape', 'infer': Reshape.infer,
                                                       'name': 'final_reshape'}),

            **result(),
        }
        edges = [
            *connect('input', '0:reshape_0'),
            *connect('reshape_0_pattern', '1:reshape_0'),
            *connect('reshape_0', '0:transpose'),
            *connect('order', '1:transpose'),
            *connect('transpose', '0:reshape_1'),
            *connect('reshape_1_pattern', '1:reshape_1'),
            *connect('reshape_1', 'output'),
        ]
        graph = build_graph(nodes, edges, nodes_with_edges_only=True, cli=Namespace())
        for node in graph.get_op_nodes():
            node['op'] = node['type']
        graph.clean_up()

        ref_nodes = {
            **regular_op_with_shaped_data('input', input_shape, {'type': 'Parameter', 'shape': int64_array(input_shape),
                                                                 'infer': Parameter.infer}),
            **regular_op_with_empty_data('depth_to_space', {'type': 'DepthToSpace', 'infer': DepthToSpaceOp.infer,
                                                            'name': 'final_reshape', 'block_size': block_size}),
            **result()
        }
        ref_edges = [*connect('input', 'depth_to_space'), *connect('depth_to_space', 'output')]
        graph_ref = build_graph(ref_nodes, ref_edges, nodes_with_edges_only=True)
        for node in graph_ref.get_op_nodes():
            node['op'] = node['type']
        graph_ref.clean_up()
        graph.graph['layout'] = 'NCHW'
        graph_ref.graph['layout'] = 'NCHW'

        return graph, graph_ref

    @pytest.mark.parametrize("input_shape, reshape_0_pattern, order, reshape_1_pattern, block_size",[
        ([1, 512, 7, 6], [1, 2, 2, 128, 7, 6], [0, 1, 4, 2, 5, 3], [1, 128, 14, 12], 2),
        ([2, 512, 7, 6], [2, 2, 2, 128, 7, 6], [0, 1, 4, 2, 5, 3], [2, 128, 14, 12], 2),
        ([1, 200, 200, 200], [1, 2, 2, 50, 200, 200], [0, 1, 4, 2, 5, 3], [1, 50, 400, 400], 2),
    ])
    def test_fusion(self, input_shape, reshape_0_pattern, order, reshape_1_pattern, block_size):
        graph, graph_ref = self.get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, block_size)
        DepthToSpaceFusion().find_and_replace_pattern(graph)
        graph.clean_up()
        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        assert flag, resp
        assert len(graph.get_op_nodes(name='final_reshape')) == 1 and \
                        graph.get_op_nodes(name='final_reshape')[0].op == 'DepthToSpace'

    @pytest.mark.parametrize("input_shape, reshape_0_pattern, order, reshape_1_pattern, group",[
        ([1, 512, 7, 6], [0, 2, 2, 128, 7, 6], [0, 1, 4, 2, 5, 3], [1, 128, 14, 12], 2),
        ([2, 512, 7, 6], [2, 2, 2, 128, 7, 6], [0, 1, 4, 2, 5, 3], [-1, 128, 14, 12], 2),
        ([1, 200, 200, 200], [1, 2, 2, 50, 200, 200], [0, 1, 4, 2, 3, 5], [1, 50, 400, 400], 2),
    ])
    def test_negative(self, input_shape, reshape_0_pattern, order, reshape_1_pattern, group):
        graph, _ = self.get_graphs(input_shape, reshape_0_pattern, order, reshape_1_pattern, group)
        graph_ref = graph.copy()
        DepthToSpaceFusion().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        assert flag, resp

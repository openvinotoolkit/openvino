# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.rank_decomposer import RankDecomposer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data

nodes = lambda output_type: {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('rank', {'op': 'Rank', 'type': None, 'output_type': output_type, 'name': 'my_rank'}),
    **result(),

    **regular_op_with_empty_data('shape', {'type': 'ShapeOf', 'output_type': output_type}),
    **regular_op_with_empty_data('rank_1D', {'type': 'ShapeOf', 'output_type': output_type}),
    **valued_const_with_data('zero', int64_array(0)),
    **regular_op_with_empty_data('rank_0D', {'type': 'Squeeze'}),
}


class TestRankDecomposerTest():

    @pytest.mark.parametrize("output_type", [np.int32, np.int64])
    def test_rank_decomposer(self, output_type):
        graph = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'rank'),
            *connect('rank', 'output'),
        ], nodes_with_edges_only=True)
        RankDecomposer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes(output_type), edges=[
            *connect('input', 'shape'),
            *connect('shape', 'rank_1D'),
            *connect('rank_1D', '0:rank_0D'),
            *connect('zero', '1:rank_0D'),
            *connect('rank_0D', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        assert flag, resp
        assert graph.get_op_nodes(type='Squeeze')[0]['name'] == 'my_rank',\
        'Name is not inherited from original node for RankDecomposer'
        print(output_type)

    def test_rank_decomposer_assertion(self):
        graph = build_graph(nodes_attrs=nodes(None), edges=[
            *connect('input', 'rank'),
            *connect('rank', 'output'),
        ], nodes_with_edges_only=True)
        with pytest.raises(AssertionError):
            RankDecomposer().find_and_replace_pattern (graph)

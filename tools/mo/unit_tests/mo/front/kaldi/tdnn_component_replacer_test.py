# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

from openvino.tools.mo.front.kaldi.tdnn_component_replacer import TdnnComponentReplacer
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, result, connect_front, const


class TestTdnnComponentReplacerTest():

    @pytest.mark.parametrize("weights, biases, time_offsets",[
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 1],),
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 1, 2, 10, 1000],),
        ([[1, 1, 1], [4, 4, 4]], [1, 2], [-1, 0]),
    ])
    def test_tdnnreplacer(self, weights, biases, time_offsets):
        def generate_offsets():
            offset_edges = []
            offset_nodes = {}

            for i, t in enumerate(time_offsets):
                offset_nodes.update(**regular_op('memoryoffset_' + str(i), {'type': None}))

                if t != 0:
                    offset_edges.append(('placeholder', 'memoryoffset_' + str(i), {'out': 0, 'in': 0}))
                    offset_edges.append(('memoryoffset_' + str(i), 'concat', {'out': 0, 'in': i}))
                else:
                    offset_edges.append(('placeholder', 'concat', {'out': 0, 'in': i}))

            return offset_nodes, offset_edges

        offset_nodes, ref_offset_edges = generate_offsets()

        nodes = {
            **offset_nodes,
            **regular_op('placeholder', {'type': 'Parameter'}),
            **regular_op('tdnncomponent', {'op': 'tdnncomponent',
                                           'weights': np.array(weights),
                                           'biases': np.array(biases),
                                           'time_offsets': np.array(time_offsets)}),
            **const('weights', np.array(weights)),
            **const('biases', np.array(biases)),
            **regular_op('concat', {'type': 'Concat', 'axis': 1}),
            **regular_op('memoryoffset_0', {'type': None}),
            **regular_op('memoryoffset_1', {'type': None}),
            **regular_op('memoryoffset_2', {'type': None}),
            **regular_op('fully_connected', {'type': 'FullyConnected'}),
            **result('result'),
        }

        graph = build_graph(nodes, [
            *connect_front('placeholder', 'tdnncomponent'),
            *connect_front('tdnncomponent', 'result')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        ref_graph = build_graph(nodes, [
            *ref_offset_edges,
            *connect_front('concat', '0:fully_connected'),
            *connect_front('weights', '1:fully_connected'),
            *connect_front('biases', '2:fully_connected'),
            *connect_front('fully_connected', 'result')
        ], nodes_with_edges_only=True)

        TdnnComponentReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'result', check_op_attrs=True)
        assert flag, resp

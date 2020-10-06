"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import unittest

from extensions.front.kaldi.replace_stats_extract_pool import StatsExtractPoolReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, connect_front, const, result

nodes = {
    **regular_op('input', {'type': 'Parameter'}),
    **regular_op('stats_ext', {'op': 'statisticsextractioncomponent', 'input_dim': 1500}),
    **regular_op('stats_pool', {'op': 'statisticspoolingcomponent'}),

    **regular_op('count', {'op': 'ReadValue'}),
    **regular_op('assign(count)', {'op': 'Assign'}),
    **const('one', int64_array([1])),
    **const('default_count', int64_array([1])),
    **regular_op('count_increment', {'op': 'Add'}),  # count = count + 1

    **const('default_Mn', int64_array([0])),
    **regular_op('Mn', {'op': 'ReadValue'}),
    **regular_op('delta_1', {'op': 'Sub'}),  # delta_1 = x - Mn
    **regular_op('delta_1/count', {'op': 'Div'}),  # delta_1 / count
    **regular_op('M(n+1)', {'op': 'Add'}),  # M(n+1) = Mn + delta_1 / count
    **regular_op('assign(M(n+1))', {'op': 'Assign'}),

    **regular_op('delta_2', {'op': 'Sub'}),  # delta_2 = x - M(n)
    **regular_op('delta_1*delta_2', {'op': 'Mul'}),  # delta_1*delta_2
    **const('default_Sn', int64_array([0])),
    **regular_op('Sn', {'op': 'ReadValue'}),
    **regular_op('S(n+1)', {'op': 'Add'}),  # S(n+1) = Sn + delta_1 * delta_2
    **regular_op('assign(S(n+1))', {'op': 'Assign'}),

    **regular_op('Var', {'op': 'Div'}),  # Variance(n) = S(n) / count
    **regular_op('concat', {'op': 'Concat'}),
    **result(),
    **result('M(n+1)_fake_output'),
    **result('S(n+1)_fake_output'),
    **result('count_fake_output'),
}

edges = [
    ('input', 'stats_ext'),
    ('stats_ext', 'stats_pool'),
    ('stats_pool', 'output')
]

edges_ref = [
    *connect_front('count', 'count_increment'),
    *connect_front('one', 'count_increment'),
    *connect_front('count_increment', 'assign(count)'),
    *connect_front('assign(count)', 'count_fake_output'),
    *connect_front('default_count', 'count'),

    *connect_front('input', '0:delta_1'),
    *connect_front('input', '0:delta_2'),
    *connect_front('default_Mn', 'Mn'),
    *connect_front('Mn', '1:delta_1'),
    *connect_front('Mn', '0:M(n+1)'),

    *connect_front('delta_1', '0:delta_1/count'),
    *connect_front('count', '1:delta_1/count'),
    *connect_front('delta_1/count', '1:M(n+1)'),  # Add from here M(n) = Mn + delta_1 / count
    *connect_front('M(n+1)', 'assign(M(n+1))'),
    *connect_front('assign(M(n+1))', 'M(n+1)_fake_output'),
    *connect_front('M(n+1)', '1:delta_2'),  # delta_2 = x - M(n)

    *connect_front('delta_1', '0:delta_1*delta_2'),
    *connect_front('delta_2', '1:delta_1*delta_2'),
    *connect_front('default_Sn', 'Sn'),
    *connect_front('Sn', '0:S(n+1)'),  # Sn from here S(n+1) = Sn + delta_1 * delta_2
    *connect_front('delta_1*delta_2', '1:S(n+1)'),  # add from here S(n) = Sn + delta_1 * delta_2
    *connect_front('S(n+1)', 'assign(S(n+1))'),
    *connect_front('assign(S(n+1))', 'S(n+1)_fake_output'),

    *connect_front('S(n+1)', '0:Var'),  # Var = S(n+1) / count
    *connect_front('count', '1:Var'),   # Var = S(n+1) / count
    *connect_front('M(n+1)', '0:concat'),
    *connect_front('Var', '1:concat'),
    *connect_front('concat', 'output'),
]


class StatsReplacerTest(unittest.TestCase):
    def test_stats_replacer(self):
        graph = build_graph(nodes_attrs=nodes, edges=edges, nodes_with_edges_only=True)
        graph.stage = 'front'

        StatsExtractPoolReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=edges_ref, nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=False)
        self.assertTrue(flag, resp)

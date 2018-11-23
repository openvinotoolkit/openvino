"""
 Copyright (c) 2018 Intel Corporation

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

from mo.middle.replacement import MiddleReplacementPattern

next_ops = ['NextIteration', 'TensorArrayWriteV3']


class DeleteSelect(MiddleReplacementPattern):
    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('Select', dict(kind='op', op='Select')),
                ('Select_data', dict(kind='data')),

                ('next_op', dict(kind='op')),
            ],
            edges=[
                ('Select', 'Select_data'),
                ('Select_data', 'next_op'),
            ],
        )

    @staticmethod
    def replace_pattern(graph, match: dict):
        if match['next_op']['op'] not in next_ops:
            return
        select = match['Select']
        assert len(select.in_nodes()) == 3

        default = select.in_node(1)
        input = select.in_node(2)

        edge_attrs = graph.get_edge_data(match['Select_data'].id, match['next_op'].id)
        graph.add_edges_from([(input.id, match['next_op'].id, edge_attrs[0])])

        graph.remove_edge(input.id, select.id)

        safe_nodes = ['next_op']
        nodes_for_remove = []
        for node in match.keys():
            if node not in safe_nodes:
                nodes_for_remove.append(match[node].id)
        graph.remove_nodes_from(nodes_for_remove)

"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.ops.embedding_bag import EmbeddingBagOffsetsSum
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, rename_nodes


class EmbeddingBagFuse(FrontReplacementSubgraph):
    enabled = True

    def run_after(self):
        from extensions.front.ExpandDimsToUnsqueeze import ExpandDimsToUnsqueeze
        from extensions.front.AttributedGatherNormalizer import AttributedGatherNormalizer
        return [ExpandDimsToUnsqueeze, AttributedGatherNormalizer]

    def pattern(self):
        return dict(
            nodes=[
                ('weights', dict(op='Const')),
                ('concat_before', dict(op='Concat')),
                ('gather_before1_1', dict(op='Gather')),
                ('unsqueeze_before1_1', dict(op='Unsqueeze')),
                ('gather_before2_1', dict(op='Gather')),
                ('unsqueeze_before2_1', dict(op='Unsqueeze')),
                ('slice1', dict(op='Slice')),
                ('gather_after1', dict(op='Gather')),
                ('reduce1', dict(op='ReduceSum')),
                ('unsqueeze_after1', dict(op='Unsqueeze')),
                ('concat_after', dict(op='Concat')),
            ],
            edges=[
                ('concat_before', 'gather_before1_1'),
                ('concat_before', 'gather_before2_1'),
                ('gather_before1_1', 'unsqueeze_before1_1'),
                ('gather_before2_1', 'unsqueeze_before2_1'),
                ('unsqueeze_before1_1', 'slice1', {'out': 0, 'in': 1}),
                ('unsqueeze_before2_1', 'slice1', {'out': 0, 'in': 2}),
                ('weights', 'gather_after1', {'out': 0, 'in': 0}),
                ('slice1', 'gather_after1', {'out': 0, 'in': 1}),
                ('gather_after1', 'reduce1'),
                ('reduce1', 'unsqueeze_after1'),
                ('unsqueeze_after1', 'concat_after'),
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        concat_before = match['concat_before']
        gather_after1 = match['gather_after1']
        slice1 = match['slice1']
        concat_after = match['concat_after']
        weights_node = gather_after1.in_port(0).get_source().node
        gather_after_axis = gather_after1.in_port(2).get_source().node.soft_get('value')
        for dst_port in weights_node.out_port(0).get_destinations():
            node = dst_port.node
            if node.op == 'Gather':
                # validate that all Gathers have same axis
                if node.in_port(2).get_source().node.soft_get('value') != gather_after_axis:
                    return
                dst_port.disconnect()
        indices_node = slice1.in_port(0).get_source().node
        slice_axis = slice1.in_port(3).get_source().node.soft_get('value')
        for dst_port in indices_node.out_port(0).get_destinations():
            node = dst_port.node
            if node.op == 'Slice':
                # validate that all Slices have same axis
                if node.in_port(3).get_source().node.soft_get('value') != slice_axis:
                    return
                dst_port.disconnect()
        emb_bag = EmbeddingBagOffsetsSum(graph, {}).create_node()
        weights_node.out_port(0).connect(emb_bag.in_port(0))
        indices_node.out_port(0).connect(emb_bag.in_port(1))
        concat_before.in_port(0).get_connection().set_destination(emb_bag.in_port(2))
        concat_after.out_port(0).get_connection().set_source(emb_bag.out_port(0))
        concat_name = concat_after.soft_get('name', concat_after.id)
        rename_nodes([(concat_after, concat_name + '/TBD'), (emb_bag, concat_name)])

        # remove this sub-graph since a lot of matchings will be obsolete
        graph.remove_nodes_from(graph.dfs(concat_before.id, set()))

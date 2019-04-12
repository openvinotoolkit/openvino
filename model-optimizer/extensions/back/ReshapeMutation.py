"""
 Copyright (c) 2019 Intel Corporation

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
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import for_each_sub_graph_recursively


class ReshapeMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op', 'type': 'Reshape'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape = match['reshape']
        if hasattr(reshape, 'dim') and reshape.dim is not None:
            reshape_inputs = reshape.in_nodes()
            value = np.array(reshape.dim)
            shape = np.array(value.shape)
            del reshape.graph.node[reshape.id]['dim']

            if 1 in reshape_inputs:
                reshape_inputs[1].value = value
                reshape_inputs[1].shape = shape
            else:
                const_id = graph.unique_id(reshape.id + '/DimData')
                graph.add_node(const_id,
                               **{'kind': 'data', 'value': value, 'shape': shape, 'name': reshape.id + '/DimData'})
                graph.add_edge(const_id, reshape.id, **{'in': 1})


class DisableReshapeMutationInTensorIterator(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [ReshapeMutation]

    @staticmethod
    def add_supported_attrs_to_node(node: Node, params: list):
        node.graph.node[node.id].update({
            'IE': [(
                'layer',
                [('id', lambda node: node.node), 'name', 'precision', 'type'],
                [
                    ('data', params, []),
                    '@ports',
                    '@consts'])]
        })

    def reshapes_with_two_inputs_to_reshape_with_dim(self, graph: Graph):
        reshapes = graph.get_op_nodes(op='Reshape')

        for reshape in reshapes:
            in_nodes = reshape.in_nodes()

            if len(in_nodes) == 1:
                continue
            assert len(in_nodes) == 2, "Reshape operation should have 2 inputs or 1 input and `dim` attribute"

            reshape['dim'] = reshape.in_port(1).get_connection().data.get_value()
            reshape.in_port(1).disconnect()

            params = [('dim', lambda node: ','.join(map(str, node['dim'])))]
            self.add_supported_attrs_to_node(reshape, params)

    def find_and_replace_pattern(self, graph: Graph):
        for_each_sub_graph_recursively(graph, self.reshapes_with_two_inputs_to_reshape_with_dim)

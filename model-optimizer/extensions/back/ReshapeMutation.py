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
from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import for_each_sub_graph_recursively


class ReshapeMutation(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    def run_before(self):
        return [ForceStrictPrecision]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        reshape = match['reshape']

        if reshape.soft_get('type') == 'Reshape':
            if graph.graph['cmd_params'].generate_experimental_IR_V10:
                reshape['force_precision_in_ports'] = {1: 'int64'}
            else:
                reshape['force_precision_in_ports'] = {1: 'int32'}


class DisableReshapeMutationInTensorIterator(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        return [ReshapeMutation]

    @staticmethod
    def add_supported_attrs_to_node(node: Node, params: list):
        node.graph.node[node.id].update({
            'IE': [(
                'layer',
                [('id', lambda node: node.node), 'name', 'precision', 'type', 'version'],
                [
                    ('data', params, []),
                    '@ports',
                    '@consts'])]
        })

    def reshapes_with_two_inputs_to_reshape_with_dim(self, graph: Graph):
        reshapes = graph.get_op_nodes(op='Reshape')

        for reshape in reshapes:
            in_nodes = reshape.in_nodes()

            assert len(in_nodes) == 2
            reshape['dim'] = reshape.in_port(1).data.get_value()

            reshape.in_port(1).disconnect()

            params = [('dim', lambda node: ','.join(map(str, node['dim'])))]
            self.add_supported_attrs_to_node(reshape, params)

    def find_and_replace_pattern(self, graph: Graph):
        for_each_sub_graph_recursively(graph, self.reshapes_with_two_inputs_to_reshape_with_dim)

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

from extensions.back.ForceStrictPrecision import ForceStrictPrecision
from extensions.back.OptimizeTransposeReshapeSequence import OptimizeTransposeReshapeSequence
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import for_each_sub_graph_recursively
from mo.ops.const import Const
from mo.ops.reshape import Reshape


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
        # TODO: WA for Caffe Alibaba model
        if reshape.has_and_set('reinterp_shape') or reshape.soft_get('type') == 'Reshape':
            if graph.graph['cmd_params'].generate_experimental_IR_V10:
                reshape['force_precision_in_ports'] = {1: 'int64'}
            else:
                reshape['force_precision_in_ports'] = {1: 'int32'}


class FlattenMutation(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True
    run_not_recursively = True

    def run_before(self):
        return [ForceStrictPrecision]

    def run_after(self):
        return [OptimizeTransposeReshapeSequence]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op', 'type': 'Flatten'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        flatten = match['reshape']
        output_shape = np.copy(flatten.out_port(0).data.get_shape())
        output_shape[0] = 0

        reshape = Reshape(graph, dict(name=flatten.id)).create_node()
        dim = Const(graph, dict(name=flatten.id + '/DimData', value=output_shape)).create_node()

        flatten.in_port(0).get_connection().set_destination(reshape.in_port(0))
        dim.out_port(0).connect(reshape.in_port(1))
        flatten.out_port(0).get_connection().set_source(reshape.out_port(0))
        reshape['force_precision_in_ports'] = {1: 'int64'}



class DisableReshapeMutationInTensorIterator(BackReplacementPattern):
    enabled = True
    force_clean_up = True
    run_not_recursively = True

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

            assert len(in_nodes) == 2
            reshape['dim'] = reshape.in_port(1).data.get_value()

            reshape.in_port(1).disconnect()

            params = [('dim', lambda node: ','.join(map(str, node['dim'])))]
            self.add_supported_attrs_to_node(reshape, params)

    def find_and_replace_pattern(self, graph: Graph):
        for_each_sub_graph_recursively(graph, self.reshapes_with_two_inputs_to_reshape_with_dim)

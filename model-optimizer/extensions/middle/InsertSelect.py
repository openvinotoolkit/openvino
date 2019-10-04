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

from extensions.ops.select import Select
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.pattern_match import find_pattern_matches, inverse_dict
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.memory import Memory
from mo.ops.result import Result
from mo.utils.error import Error
from mo.utils.graph import invert_sub_graph_between_nodes


class AddSelectBeforeMemoryNodePattern(MiddleReplacementPattern):
    """
    Add Select before saving state with Memory to avoid garbage saving
    """
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[('op', dict(op='Memory', index=0))],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']

        if node.name == 'iteration_number_out':
            return

        # calculate length of context when state of inference becomes meaningful
        inputs = []
        for n in graph.get_op_nodes(**{'op': 'Parameter'}):
            inputs.append(n)

        in_nodes = []
        for inp in inputs:
            for ins in inp.out_port(0).get_destinations():
                in_nodes.append(ins.node.name)

        context_len = 1
        try:
            subgraph = invert_sub_graph_between_nodes(graph, [node.in_port(0).get_source().node.name], in_nodes)
        except Error:
            return

        for n in subgraph:
            n_node = Node(graph, n)
            if n_node.kind == 'op' and n_node.op == 'Splice':
                context_len += len(n_node.context) - 1

        if context_len == 1:
            return

        in_node_port = node.in_port(0).get_source()
        in_node_shape = node.in_port(0).data.get_shape()
        node.in_port(0).disconnect()

        # add Select before saving state to avoid saving garbage
        select_node = Select(graph, {'name': 'select_' + node.name}).create_node()
        zero_else = Const(graph, {'name': 'zero_else', 'value': np.zeros(in_node_shape)}).create_node()
        select_node.in_port(1).connect(in_node_port)
        select_node.in_port(2).connect(zero_else.out_port(0))

        # check if we have already appropriate iteration counter
        existing_counters = find_pattern_matches(graph, nodes=[('mem_in', dict(op='Memory', index=1,
                                                                               shape=int64_array([context_len]))),
                                                               ('mem_in_data', dict()),
                                                               ('crop_mem_in', dict(op='Crop', axis=int64_array([1]),
                                                                                    offset=int64_array([1]),
                                                                                    dim=int64_array([context_len-1]))),
                                                               ('crop_mem_in_data', dict()),
                                                               ('concat', dict(op='Concat', axis=1)),
                                                               ('concat_data', dict()),
                                                               ('const_1', dict(op='Const')),
                                                               ('const_1_data', dict()),
                                                               ('mem_out', dict(op='Memory', index=0,
                                                                                shape=int64_array([context_len]))),
                                                               ('crop_out', dict(op='Crop', axis=int64_array([1]),
                                                                                 offset=int64_array([0]),
                                                                                 dim=int64_array([1]))),
                                                               ('crop_out_data', dict()),
                                                               ('select', dict(op='Select'))
                                                               ],
                                                 edges=[('mem_in', 'mem_in_data'), ('mem_in_data', 'crop_mem_in'),
                                                        ('crop_mem_in', 'crop_mem_in_data'),
                                                        ('crop_mem_in_data', 'concat', {'in': 0}),
                                                        ('const_1', 'const_1_data'),
                                                        ('const_1_data', 'concat', {'in': 1}),
                                                        ('concat', 'concat_data'), ('concat_data', 'mem_out'),
                                                        ('concat_data', 'crop_out'), ('crop_out', 'crop_out_data'),
                                                        ('crop_out_data', 'select')])
        counter_match = next(existing_counters, None)
        if counter_match is not None:
            input_port = Node(graph, inverse_dict(counter_match)['crop_out']).out_port(0)
        else:
            mem_out = Memory(graph, {'name': 'iteration_number', 'size': 2,
                                     'index': 1, 'id': 'iteration_'+node.name,
                                     'shape': int64_array([context_len]),
                                     'force_precision': 'I32'}).create_node()
            cut_first = Crop(graph, {'name': 'cut_first', 'axis': int64_array([1]),
                                     'offset': int64_array([1]), 'dim': int64_array([context_len-1]),
                                     'force_precision': 'I32'}).create_node()
            cut_first.in_port(0).connect(mem_out.out_port(0))
            ones = Const(graph, {'name': 'ones', 'value': np.ones([1, 1], dtype=np.int64),
                                 'force_precision': 'I32'}).create_node()
            concat = Concat(graph, {'name': 'concat_ones', 'in_ports_count': 2, 'axis': 1,
                                    'force_precision': 'I32'}).create_node()
            concat.in_port(0).connect(cut_first.out_port(0))
            concat.in_port(1).connect(ones.out_port(0))
            mem_in = Memory(graph, {'name': 'iteration_number_out', 'size': 2,
                                    'index': 0, 'id': 'iteration_'+node.name,
                                    'shape': int64_array([context_len]),
                                    'force_precision': 'I32'}).create_node()
            mem_in.in_port(0).connect(concat.out_port(0))
            res = Result(graph, {}).create_node()
            mem_in.out_port(0).connect(res.in_port(0))
            cut_last = Crop(graph, {'name': 'cut_last', 'axis': int64_array([1]),
                                    'offset': int64_array([0]), 'dim': int64_array([1]),
                                    'force_precision': 'I32'}).create_node()
            cut_last.in_port(0).connect(concat.out_port(0))
            input_port = cut_last.out_port(0)

        select_node.in_port(0).connect(input_port)
        select_node.out_port(0).connect(node.in_port(0))
        select_node.out_port(0).data.set_shape(in_node_shape)

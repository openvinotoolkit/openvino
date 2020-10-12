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

from extensions.ops.MatMul import FullyConnected
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node
from mo.graph.graph import rename_nodes
from mo.ops.concat import Concat
from mo.ops.memoryoffset import MemoryOffset


class TdnnComponentReplacer(FrontReplacementPattern):
    '''
    Expand TdnnComponent into MemoryOffsets, Concat and FullyConected nodes

    BEFORE:
                          placeholder
                              |
                      TdnnComponent('time_offsets': t1, t2,... tk)
                              |
    _______________________________________________________________

    AFTER:
                          placeholder
            __________________|___________________________
           /                  |              \            \
   MemoryOffset(t1)     MemoryOffset(t2)    ...    MemoryOffset(tk)
          \_____________ _____|______________/____________/
                           Concat
                             |
                        FullyConnected
                             |
    '''
    enabled = True
    run_not_recursively = True

    def run_before(self):
        from extensions.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
        return [MemoryOffsetAdjustment]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='tdnncomponent'):
            self.replace_tdnn(graph, node)

    def replace_tdnn(self, graph: Graph, tdnn_node: Node):
        tdnn_name = tdnn_node.soft_get('name', tdnn_node.id)

        concat_node = Concat(graph, {'axis': 1}).create_node()
        rename_nodes([(tdnn_node, tdnn_name + '/to_be_removed'), (concat_node, tdnn_name)])

        for offset_ind, t in enumerate(tdnn_node['time_offsets']):
            concat_node.add_input_port(offset_ind)
            if t != 0:
                memory_name = tdnn_name + '/MemoryOffset/' + str(abs(t))
                memoryoffset_node = MemoryOffset(graph, {'name': memory_name, 't': t,
                                                         'pair_name': memory_name + '_out',
                                                         'has_default': False, 'splitted': False}).create_node()

                tdnn_node.in_port(0).get_source().connect(memoryoffset_node.in_port(0))
                memoryoffset_node.out_port(0).connect(concat_node.in_port(offset_ind))
            else:
                # 0 time delay is not allowed in IE, it's meaningless
                # if time offset is 0 then connect input of tdnncomponent directly to Concat without memoryoffset
                tdnn_node.in_port(0).get_source().connect(concat_node.in_port(offset_ind))

        weights = tdnn_node['weights']
        fc_inputs = {1: weights}

        bias_term = False
        if tdnn_node.has_valid('biases'):
            assert len(tdnn_node['biases']) == weights.shape[0]
            fc_inputs.update({2: tdnn_node['biases']})
            bias_term = True

        fc_node = create_op_with_const_inputs(graph, FullyConnected, fc_inputs,
                                              {'name': tdnn_name + '/FC', 'out-size': weights.shape[0],
                                               'transpose_weights': True, 'bias_term': bias_term})

        concat_node.out_port(0).connect(fc_node.in_port(0))
        tdnn_node.in_port(0).disconnect()
        tdnn_node.out_port(0).get_connection().set_source(fc_node.out_port(0))

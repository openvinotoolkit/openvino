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

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.concat import Concat
from extensions.ops.MatMul import FullyConnected


class TdnnComponentReplacer(FrontReplacementPattern):
    '''
    Replace TdnnComponent with MemoryOffsets
    '''
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        from extensions.front.kaldi.split_memoryoffsets import SplitMemoryOffsets
        return [SplitMemoryOffsets]

    def pattern(self):
        return dict(
            nodes=[
                ('tdnncomponent', dict(kind='op', op='tdnncomponent'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        tdnn_node: Node = match['tdnncomponent']
        tdnn_name = tdnn_node.name
        concat_node = Concat(graph, {'name': tdnn_name + '_concat', 'axis': 0}).create_node()

        for i, t in enumerate(tdnn_node['time_offsets']):
            memory_name = tdnn_name + '_memoryoffset_' + str(t)
            memoryoffset_node = MemoryOffset(graph, {'name': memory_name, 't': t,
                                 'pair_name': memory_name + '_out',
                                 'has_default': False}).create_node()
            tdnn_node.in_port(0).get_source().connect(memoryoffset_node.in_port(0))

            concat_node.add_input_port(i)
            memoryoffset_node.out_port(0).connect(concat_node.in_port(i))

        fc_layer = FullyConnected(graph, {'name': tdnn_name + '_fc',
                                          'out-size': None, 'transpose_weights': False,
                                          'bias_term': False}).create_node()
        concat_node.out_port(0).connect(fc_layer.in_port(0))
        tdnn_node.in_port(0).disconnect()
        tdnn_node.out_port(0).get_connection().set_source(fc_layer.out_port(0))
        print('hey')
        pass

        # add fully to concat
        # add const nodes to fully
        # set source of tdnns output to fully connected out


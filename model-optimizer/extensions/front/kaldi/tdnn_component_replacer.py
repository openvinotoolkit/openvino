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
from mo.graph.graph import Graph
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.result import Result


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
        tdnn_node = match['tdnncomponent']
        tdnn_name = tdnn_node.name

        for i, t in enumerate(tdnn_node['time_offsets']):
            memory_name = tdnn_name + '_memoryoffset_' + str(t)
            MemoryOffset(graph, {'name': memory_name, 't': t,
                                 'pair_name': memory_name + '_out',
                                 'has_default': False}).create_node()
        pass
        # concat_op = Concat(graph, dict(axis=3)).create_node([crop_batch_node, crop_coordinates_node], dict(name='batch_and_coords', nchw_layout=True))
        # connect all offsets to concat

        # add convolution to concat
        # add const nodes to convolution
        # reconnect concat to out of tdnn

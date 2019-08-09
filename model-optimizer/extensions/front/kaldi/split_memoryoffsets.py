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
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.result import Result


class SplitMemoryOffsets(FrontReplacementPattern):
    '''
    Split MemoryOffsets in 2 parts to cut cycles
    '''
    enabled = True
    run_not_recursively = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def pattern(self):
        return dict(
            nodes=[
                ('mem_offset', dict(kind='op', op='MemoryOffset', splitted=False))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        offset_node = match['mem_offset']
        paired_node = MemoryOffset(graph, {'name': offset_node.pair_name, 'splitted': True, 'pair_name': offset_node.id,
                                           't': offset_node.t, 'has_default': offset_node.has_default}).create_node()
        offset_node['splitted'] = True
        offset_node.out_port(0).get_connection().set_source(paired_node.out_port(0))
        res_node = Result(graph, {'name': offset_node.id+"_output"}).create_node()
        offset_node.out_port(0).connect(res_node.in_port(0))

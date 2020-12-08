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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.result import Result


class SplitTdnnMemoryOffset(MiddleReplacementPattern):
    '''
    Splits MemoryOffsets in TDNN blocks into 2 parts. These parts then will be converted to ReadValue and Assign.
    '''
    enabled = True
    run_not_recursively = True

    def run_before(self):
        from extensions.middle.ReplaceMemoryOffsetWithSplice import ReplaceMemoryOffsetWithMemoryNodePattern, ReplaceMemoryOffsetNodePattern
        return [ReplaceMemoryOffsetNodePattern, ReplaceMemoryOffsetWithMemoryNodePattern]

    def find_and_replace_pattern(self, graph: Graph):
        for offset_node in graph.get_op_nodes(op='MemoryOffset', splitted=False):
            paired_node = MemoryOffset(graph, {'name': offset_node.pair_name, 'splitted': True, 'pair_name': offset_node.id,
                                               't': offset_node.t, 'has_default': offset_node.has_default}).create_node()
            offset_node['splitted'] = True
            offset_node.out_port(0).get_connection().set_source(paired_node.out_port(0))
            res_node = Result(graph, {'name': offset_node.id + "_output"}).create_node()
            offset_node.out_port(0).connect(res_node.in_port(0))

            # If 'element_size' is previously copied from Parameter of from node with defined dim
            if offset_node.has_valid('element_size'):
                paired_node['element_size'] = offset_node['element_size']
            # Copy shape from previous node. Typically (but not always) for TDNN blocks this is the case
            else:
                paired_node['element_size'] = offset_node.in_port(0).data.get_shape()[1]

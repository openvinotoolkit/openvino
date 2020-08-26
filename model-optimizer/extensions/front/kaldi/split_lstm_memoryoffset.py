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
import networkx as nx

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.ops.memoryoffset import MemoryOffset
from mo.ops.result import Result
from mo.utils.error import Error


class SplitLstmMemoryOffset(FrontReplacementSubgraph):
    """
    Splits MemoryOffsets in LSTM blocks into 2 parts. These parts then will be converted to ReadValue and Assign.

    Splitting complicates shape inference but MemoryOffsets in LSTM blocks are cycled and, in order to make topological
    sort possible during shape inference, they are splitted earlier on the front phase. In contrast, MemoryOffsets in
    TDNN blocks are not cycled, so they will be splitted after shape infer on the middle.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def split_offset(self, offset_node):
        paired_node = MemoryOffset(offset_node.graph, {'name': offset_node.pair_name, 'splitted': True,
                                           'pair_name': offset_node.id,
                                           'element_size': offset_node['element_size'],
                                           't': offset_node.t,
                                           'has_default': offset_node.has_default}).create_node()
        offset_node['splitted'] = True
        offset_node.out_port(0).get_connection().set_source(paired_node.out_port(0))
        res_node = Result(offset_node.graph, {'name': offset_node.id + "_output"}).create_node()
        offset_node.out_port(0).connect(res_node.in_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for offset_node in graph.get_op_nodes(op='MemoryOffset'):
            try:
                # if graph contains LSTM block with cycle, split MemoryOffset to enable shape infer
                nx.find_cycle(graph, offset_node.id)
                assert offset_node.has_valid('element_size')
                self.split_offset(offset_node)
            except nx.NetworkXNoCycle as e:
                # MemoryOffset node is not in a recursive LSTM block -- no splitting is needed
                return
            except AssertionError as e:
                raise Error("MemoryOffset node in a LSTM block has no element_size. Out shape can not be inferred")

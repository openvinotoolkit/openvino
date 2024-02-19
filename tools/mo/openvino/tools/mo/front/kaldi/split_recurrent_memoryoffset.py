# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.memoryoffset import MemoryOffset
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.graph import Node


class SplitRecurrentMemoryOffset(FrontReplacementSubgraph):
    """
    Splits MemoryOffsets in recurrent blocks (typically LSTM blocks) into 2 parts.

    These parts then will be converted to ReadValue and Assign. Splitting complicates shape inference but
    MemoryOffsets in recurrent blocks are cycled and, in order to make topological sort possible
    during shape inference, they are splitted earlier on the front phase. In contrast,
    MemoryOffsets in TDNN blocks are not cycled, so they will be splitted after shape infer on the middle.
    Now only LSTM blocks with MemoryOffset are present.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    @staticmethod
    def split_offset(offset_node: Node):
        paired_node = MemoryOffset(offset_node.graph, {'name': offset_node.pair_name, 'splitted': True,
                                                       'pair_name': offset_node.id,
                                                       'element_size': offset_node['element_size'],
                                                       't': offset_node.t,
                                                       'has_default': offset_node.has_default}).create_node()
        offset_node['splitted'] = True
        offset_node.out_port(0).get_connection().set_source(paired_node.out_port(0))
        res_node = Result(offset_node.graph, {'name': offset_node.id + '_output'}).create_node()
        offset_node.out_port(0).connect(res_node.in_port(0))

    def find_and_replace_pattern(self, graph: Graph):
        for offset_node in graph.get_op_nodes(op='MemoryOffset', splitted=False):
            try:
                # if graph contains recurrent block -> split MemoryOffset to enable shape infer
                nx.find_cycle(graph, offset_node.id)
            except nx.NetworkXNoCycle as e:
                # MemoryOffset node is not in a recurrent block -- no splitting is needed
                return

            # check that node has information for future partial infer
            # element_size is set in loader based on dimensions of previous layer from original Kaldi model
            if not offset_node.has_valid('element_size'):
                # check if previous layer contains information about its shape in out-size
                # out-size is set in extractor of some nodes like affinecomponent based on weight's size
                if offset_node.in_port(0).get_source().node.has_valid('out-size'):
                    offset_node['element_size'] = int64_array([1, offset_node.in_port(0).get_source().node['out-size']])
                else:
                    raise Error("In a recurrent block 'element_size' for node {} is not set".format(offset_node.id))
            SplitRecurrentMemoryOffset.split_offset(offset_node)

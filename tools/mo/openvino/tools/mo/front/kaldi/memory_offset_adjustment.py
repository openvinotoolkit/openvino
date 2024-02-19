# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.memoryoffset import MemoryOffset


def find_max_frame_time(node: Node):
    in_frame_time_max = 0
    should_align = False
    for inp in node.in_ports():
        if node.in_port(inp).disconnected():
            continue
        in_node = node.in_port(inp).get_source().node
        if in_node.frame_time > in_frame_time_max:
            in_frame_time_max = in_node.frame_time

    if in_frame_time_max == 0:
        return in_frame_time_max, False

    for inp in node.in_ports():
        if node.in_port(inp).disconnected():
            continue
        if in_frame_time_max != node.in_port(inp).get_source().node.frame_time:
            should_align = True
            break

    return in_frame_time_max, should_align


def align_frame_time(graph: Graph, node: Node, frame_time_max):
    for inp in node.in_ports():
        if node.in_port(inp).disconnected():
            continue
        in_node = node.in_port(inp).get_source().node
        in_node_out_port = node.in_port(inp).get_source()
        in_port = node.in_port(inp)
        # Adding MemoryOffset for Const does not make sense
        if in_node.frame_time < frame_time_max and in_node.op != 'Const':
            # Change existing MemoryOffset to avoid adding new one
            if in_node.op == 'MemoryOffset':
                in_node.t = in_node.frame_time - frame_time_max
                in_node.frame_time = in_node.t
            else:
                mem_name = graph.unique_id("align_" + node.id)
                memory_align = MemoryOffset(graph, attrs={'id': mem_name,
                                                          'name': mem_name,
                                                          'pair_name': mem_name + "_pair",
                                                          't': in_node.frame_time - frame_time_max,
                                                          'splitted': False}).create_node()
                # add element_size for MemoryOffset after Parameter for infer
                if in_node.op == 'Parameter':
                    memory_align['element_size'] = in_node.shape

                memory_align.in_port(0).get_connection().set_source(in_node_out_port)
                in_port.get_connection().set_source(memory_align.out_port(0))
                memory_align['frame_time'] = memory_align.t
        # remove MemoryOffset with maximum delay
        elif in_node.frame_time == frame_time_max and in_node.op == 'MemoryOffset':
            in_node_out_port.get_connection().set_source(in_node.in_port(0).get_source())
            graph.remove_node(in_node.id)


class MemoryOffsetAdjustment(FrontReplacementSubgraph):
    r"""
    Pass used to fix wrong results in the following situation:
                              input
                              |   \
                            ...   ...
                             |       \
                    MemoryOffset(k)   \
                             |        |
                             ...      |
                              \      |
                               \     |
                               Concat
    In Left branch we have MemoryOffset with k > 0 so we wait until kth frame will be calculated. In right branch
    we have no such offsets. As result we Concat (or use in any calculations with more than 1 input) kth frame from
    left branch and 0th from right branch. So we need to add synchronization before Concat node. it can be done with
    MemoryOffset(k) inserted before Concat.

    Main idea of this change that when we found memoryOffset with t>0 we should re-calculate all delays relative to this
    t.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'kaldi']

    def run_before(self):
        # transformation can't work with splitted MemoryOffsets
        from openvino.tools.mo.front.kaldi.split_recurrent_memoryoffset import SplitRecurrentMemoryOffset
        return [SplitRecurrentMemoryOffset]

    def find_and_replace_pattern(self, graph: Graph):
        should_continue = False
        for n in graph:
            if Node(graph, n).op == 'MemoryOffset' and Node(graph, n).t > 0:
                should_continue = True
                break

        if not should_continue:
            return

        try:
            nodes = list(nx.topological_sort(graph))
        except:
            return

        nx.set_node_attributes(G=graph, name='frame_time', values=-1)

        for n in nodes:
            node = Node(graph, n)

            # calculate frame_time (delay) that was not calculated
            if node.frame_time < 0:
                # MemoryOffset with t>0 increases frame delay
                if node.op == "MemoryOffset":
                    node.frame_time = node.in_port(0).get_source().node.frame_time + node.t
                # for node with several inputs frame_time = maximum of delays from branches
                # other branches should be synced by adding MemoryOffset(branch frame_time  - max)
                # After that MemoryOffset with maximum delay should be deleted (t becomes 0)
                elif len(node.in_edges()) > 1:
                    # find out maximum of delay and check that we have at least one branch with another delay
                    in_frame_time_max, should_align = find_max_frame_time(node)
                    if should_align:
                        align_frame_time(graph, node, in_frame_time_max)
                    node.frame_time = in_frame_time_max
                elif len(node.in_edges()) == 1:
                    node.frame_time = node.in_port(0).get_source().node.frame_time
                else:
                    # for all input nodes (without inputs) frame_time is 0
                    node.frame_time = 0

        for n in graph:
            node = Node(graph, n)
            if 'frame_time' in node:
                del node['frame_time']

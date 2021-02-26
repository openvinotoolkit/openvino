"""
 Copyright (C) 2018-2021 Intel Corporation

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
from collections import deque
from typing import Set, List

from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


class MarkShapeOfSubgraphDataType(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PostMiddleStart
        return [PostMiddleStart]

    def run_before(self):
        from extensions.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
        return [MarkSubGraphsWithCorrectLayout]

    def find_and_replace_pattern(self, graph: Graph):
        bfs_search_apply_on_shapeof_subgraph_nodes(graph.get_op_nodes(op='ShapeOf'), action=mark_nodes)


def mark_nodes(node: Node):
    node['in_shape_subgraph'] = True
    for out_port in node.out_ports().values():
        if not out_port.disconnected():
            node.out_node(out_port.idx)['correct_data_type'] = True
            node.out_node(out_port.idx)['in_shape_subgraph'] = True


def get_next_op_in_nodes(node: Node, include_in_port_0=True) -> Set[Node]:
    in_nodes = set()
    for in_port in node.in_ports().values():
        if not in_port.disconnected():
            if include_in_port_0 or not in_port.idx == 0:  # need to skip optional not connected inputs
                in_nodes.add(in_port.get_source().node)
    return in_nodes


def get_next_op_out_nodes(node: Node) -> Set[Node]:
    out_nodes = set()
    for out_port in node.out_ports().values():
        if not out_port.disconnected():
            dst_nodes = [dst.node for dst in out_port.get_destinations()]
        out_nodes.update(dst_nodes)
    return out_nodes


def bfs_search_apply_on_shapeof_subgraph_nodes(initial_nodes: List[Node], action: callable = None):
    """
    One more bfs since it is complicated to reuse other existing BFS routines. Action will be taken not
    on all visited nodes but only on those that are inside ShapeOf subgraph.
    :param initial_nodes:
    :param action:
    """
    visited = set(initial_nodes)
    end_point_nodes = set()

    deque_of_nodes = deque()
    for node in initial_nodes:
        deque_of_nodes.extend(get_next_op_out_nodes(node))

    while len(deque_of_nodes):
        node = deque_of_nodes.popleft()
        if node in visited:
            continue
        if any([out_port.data.get_value() is None for out_port in node.out_ports().values()]):
            end_point_nodes.add(node)
            visited.add(node)
            continue

        next_in_nodes = get_next_op_in_nodes(node)
        next_out_nodes = get_next_op_out_nodes(node)
        deque_of_nodes.extend(next_out_nodes)
        deque_of_nodes.extend(next_in_nodes)
        visited.add(node)
        if action is not None:
            action(node)

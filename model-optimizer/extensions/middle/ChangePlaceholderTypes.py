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

import logging as log

from mo.graph.graph import Graph, Node
from mo.middle.passes.fusing.helpers import get_next_operation
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class ChangePlaceholderTypes(MiddleReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']
    force_clean_up = True

    def run_after(self):
        return []

    def run_before(self):
        from extensions.middle.ScaleInput import ScaleInput
        return [ScaleInput]

    @staticmethod
    def change_node_type(node: Node, new_type: type):
        node.graph.node[node.id]['pb'].attr['dtype'].type = new_type

    @staticmethod
    def is_node_casts_to_float(node: Node):
        from tensorflow.core.framework import types_pb2 as tf_types  # pylint: disable=no-name-in-module
        attrs = node.graph.node[node.id]
        return 'pb' in attrs and attrs['pb'].op == 'Cast' and attrs['pb'].attr['DstT'].type == tf_types.DT_FLOAT

    @staticmethod
    def remove_node_preserving_edges(pl_node: Node, nodes: list):
        graph = pl_node.graph
        pl_node_data = pl_node.out_node()

        # Disconnect Placeholder data node from Cast nodes
        for out_node in pl_node.out_node().out_nodes():
            graph.remove_edge(pl_node_data.id, out_node.id)

        # Move edges from Cast data nodes to Placeholder data node
        for cast_node in nodes:
            # it is necessary to create a list from the result of function "graph.out_edges()" because we modify
            # the graph during iteration over the list. networkx version 2.1 raises error without creating a list
            for u, v, d in list(graph.out_edges(cast_node.out_node().id, data=True)):
                graph.remove_edge(u, v)
                graph.add_edges_from([(pl_node_data.id, v, d)])

    @staticmethod
    def is_node_gather(node: Node):
        attrs = node.graph.node[node.id]
        return 'pb' in attrs and attrs['pb'].op == 'GatherV2' and attrs['precision'] == 'FP32'

    def find_and_replace_pattern(self, graph: Graph):
        from tensorflow.core.framework import types_pb2 as tf_types  # pylint: disable=no-name-in-module
        for node_name, node_attrs in list(graph.nodes(data=True)):
            node = Node(graph, node_name)
            pb = node_attrs.get('pb')
            if pb is not None and pb.op == 'Parameter' and pb.attr['dtype'].type != tf_types.DT_FLOAT:
                log.info('Placeholder "{}" has type that is different from DT_FLOAT'.format(node_name))
                next_ops = get_next_operation(node)
                # check that all output nodes are nodes of type 'ToFloat'
                if all([ChangePlaceholderTypes.is_node_casts_to_float(op) and
                        len(op.in_nodes()) == 1 for op in next_ops]):
                    ChangePlaceholderTypes.change_node_type(node, tf_types.DT_FLOAT)
                    ChangePlaceholderTypes.remove_node_preserving_edges(node, next_ops)  # remove 'Cast' nodes

                elif all([ChangePlaceholderTypes.is_node_gather(op) for op in next_ops] for op in next_ops):
                    ChangePlaceholderTypes.change_node_type(node, tf_types.DT_FLOAT)

                else:
                    raise Error(
                        ('Cannot convert type of placeholder "{}" because not all of its outputs are "Cast" to float '
                         'operations: {}. ' +
                         refer_to_faq_msg(49)),
                        node.soft_get('name'),
                        [op.soft_get('name') for op in next_ops]
                    )

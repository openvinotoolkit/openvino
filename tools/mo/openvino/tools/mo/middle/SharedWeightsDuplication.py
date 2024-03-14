# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.op import Op


class SharedWeightsDuplication(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.CheckForCycle import CheckForCycle
        return [CheckForCycle]

    def run_before(self):
        from openvino.tools.mo.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        """
        This function finds all const data nodes that have more that one consumer and then duplicate them
        """
        data_nodes = [Node(graph, id) for id in graph.nodes() if Node(graph, id).soft_get('kind') == 'data']
        for node in data_nodes:
            # Check that node has const values and more than one consumer
            if len(node.in_nodes()) and node.in_node().soft_get('type') == 'Const' and len(node.out_nodes()) > 1 and \
                            node.value is not None:
                # Here we delete all edges between base node and it's consumers (except first), and then duplicate this
                # node to connect with other consumers
                for v, d in node.get_outputs():
                    out_node = Node(graph, v)
                    e_attrs = d
                    graph.remove_edge(node.id, out_node.id)
                    data = Op.create_input_data_node(graph, "Copy_{}".format(node.id), mo_array(node.value),
                                                     graph.node[node.id])

                    graph.add_edges_from([(data.id, out_node.id, e_attrs)])


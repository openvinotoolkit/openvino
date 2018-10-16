"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np

from mo.graph.graph import Node
from mo.ops.op import Op
from mo.utils.error import Error


def duplicate_shared_weights(graph: nx.MultiDiGraph):
    """
    This function finds all const data nodes that have more that one consumer and then duplicate them
    """
    data_nodes = [Node(graph, id) for id in graph.nodes() if Node(graph, id).soft_get('kind') == 'data']
    for node in data_nodes:
        # Check that node has const values and more than one consumer
        if len(node.out_nodes()) > 1 and node.value is not None:
            # Here we delete all edges between base node and it's consumers (except first), and then duplicate this
            # node to connect with other consumers
            while len(node.out_nodes()) > 1:
                out_node = node.out_node(1)

                if len(graph.get_edge_data(node.id, out_node.id)) != 1:
                    raise Error('There is more than one edge from {} node to {} node.'.format(node.id, out_node.id))
                e_attrs = graph.get_edge_data(node.id, out_node.id)[0]

                graph.remove_edge(node.id, out_node.id)
                data = Op.create_input_data_node(graph, "Copy_{}".format(node.id), np.array(node.value), graph.node[node.id])

                graph.add_edges_from([(data.id, out_node.id, e_attrs)])

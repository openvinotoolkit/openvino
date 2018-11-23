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
from copy import deepcopy
from mo.front.common.layout import get_features_dim, shape_for_layout
from mo.graph.graph import Node
from mo.middle.passes.fusing.helpers import get_value_id
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import Op
from mo.ops.reshape import Reshape


class Eltwise1DInputReshape(MiddleReplacementPattern):
    """
    Inserts Reshape before 1-D input to Eltwise if another input of Eltwise is multi-dimensional tensor with the
    same feature size as 1-D input

    Replacer is useful in cases of layout change in MO (for example NHWC-> NCHW translation of TensorFlow models)

    Example:
    Eltwise Mul operation in TF multiplies Tensors by feature dimension with shapes [1,375,500,24] and [24].
    After layout change in MO Eltwise Mul have input shapes [1,24,375,500] and [24]. It is a problem (500!=24).
    We have to insert Reshape layer for Tensor with [24] shape to correspond the feature dimension of
    Tensor [1,24,375,500] shape

    change of graph.graph['layout'] may cause an issue
    change in re-layout function: convert_nhwc_to_nchw(graph) may cause an issue
    """
    enabled = True

    def run_after(self):
        return [EltwiseInputReshape]

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        layout = graph.graph['layout']
        for n in nx.topological_sort(graph):
            if 'type' in graph.node[n] and graph.node[n]['type'] == 'Eltwise' and get_value_id(Node(graph, n)) is None:
                eltwise_op_node = Node(graph, n)
                out_shape = eltwise_op_node.out_node().shape
                if 4 <= len(out_shape) <= 5:
                    out_features = out_shape[get_features_dim(layout, len(out_shape))]
                    for port, node in eltwise_op_node.in_nodes().items():
                        if len(node.shape) != len(out_shape) and len(node.shape) == 1 and out_features == node.shape[0]:
                            in_atts = deepcopy(graph.get_edge_data(node.id, n)[0])
                            graph.remove_edge(node.id, n)
                            new_shape = shape_for_layout(layout, batch=1, features=out_features, height=1, width=1,
                                                         depth=1 if len(out_shape) == 5 else None)
                            reshape_data_op = Reshape(graph, attrs={'dim': new_shape, 'name': node.id + '/Broadcast'})
                            reshape_data_node = reshape_data_op.create_node_with_data([node])
                            graph.add_edge(reshape_data_node.id, eltwise_op_node.id, **in_atts)


class EltwiseInputReshape(MiddleReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: nx.MultiDiGraph):
        data_nodes = [Node(graph, node) for node in graph.node if Node(graph, node).kind == 'data']
        for node in data_nodes:
            # Get all requested shapes for current node
            # This mapping will contain pairs like {shape:[list of consumers nodes]}
            mapping = {}
            for consumer in node.out_nodes():
                edge_attrs = graph.get_edge_data(node.id, consumer.id)[0]
                if 'new_shape' in edge_attrs:
                    if np.array_equal(edge_attrs['new_shape'], node.shape):
                        continue
                    new_shape = tuple([x for x in edge_attrs['new_shape']])
                    if not new_shape in mapping:
                        mapping.update({new_shape: [consumer]})
                    else:
                        mapping[new_shape].append(consumer)

            if node.has_valid('value'):
                # Check that requested shape are the same
                # In case if they are different, we duplicate them
                for shape_key in mapping.keys():
                    shape = list(shape_key)
                    new_value = np.reshape(node.value, shape)
                    node_copy = Op.create_input_data_node(graph, node.id + '/copy', value=np.array(new_value))
                    for consumer in mapping[shape_key]:
                        edge_attrs = graph.get_edge_data(node.id, consumer.id)[0]
                        del edge_attrs['new_shape']

                        # Remove edge from previous data node and connect new data node with its consumer
                        graph.remove_edge(node.id, consumer.id)
                        graph.add_edge(node_copy.id, consumer.id, **edge_attrs)
            else:
                # Insert Reshape layer between data node and consumer
                for shape_key in mapping.keys():
                    shape = list(shape_key)
                    reshape = Reshape(graph, attrs={'dim': shape, 'name': 'EltwiseReshapeNormalization'})
                    reshape_data = reshape.create_node_with_data(inputs=[node])

                    # Iterate over consumers and reconnect them to Reshape layer output
                    for consumer in mapping[shape_key]:
                        edge_attrs = graph.get_edge_data(node.id, consumer.id)[0]
                        del edge_attrs['new_shape']

                        # Reconnect edge from original data node to Reshape output datanode
                        graph.remove_edge(node.id, consumer.id)
                        graph.add_edge(reshape_data.id, consumer.id, **edge_attrs)
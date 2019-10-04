"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from mo.front.common.layout import get_features_dim, shape_for_layout
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
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
    enabled = False

    def run_after(self):
        return [EltwiseInputReshape]

    def find_and_replace_pattern(self, graph: Graph):
        layout = graph.graph['layout']
        for eltwise_op_node in graph.get_op_nodes(is_eltwise=True):
                out_shape = eltwise_op_node.out_port().data.get_shape()
                if 4 <= len(out_shape) <= 5:
                    out_features = out_shape[get_features_dim(layout, len(out_shape))]
                    for port, node in eltwise_op_node.in_nodes().items():
                        if len(node.shape) != len(out_shape) and len(node.shape) == 1 and out_features == node.shape[0]:
                            new_shape = shape_for_layout(layout, batch=1, features=out_features, height=1, width=1,
                                                         depth=1 if len(out_shape) == 5 else None)
                            dim_const = Const(graph, {'value': new_shape, 'name': node.id + '/Dim'}).create_node()
                            reshape_op = Reshape(graph, attrs={'dim': new_shape, 'name': node.id + '/Broadcast'}).create_node()

                            eltwise_op_node.in_port(port).get_source().node.out_port(0).get_connection().set_destination(reshape_op.in_port(0))
                            reshape_op.in_port(1).connect(dim_const.out_port(0))

                            reshape_op.out_port(0).connect(eltwise_op_node.in_port(port))


class EltwiseInputReshape(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_data_nodes():
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
                    reshape = Reshape(graph, attrs={'name': 'EltwiseReshapeNormalization'})
                    reshape_dim = Const(graph, {'value': shape}).create_node_with_data()
                    reshape_data = reshape.create_node_with_data(inputs=[node, reshape_dim])

                    # Iterate over consumers and reconnect them to Reshape layer output
                    for consumer in mapping[shape_key]:
                        edge_attrs = graph.get_edge_data(node.id, consumer.id)[0]
                        del edge_attrs['new_shape']

                        # Reconnect edge from original data node to Reshape output datanode
                        graph.remove_edge(node.id, consumer.id)
                        graph.add_edge(reshape_data.id, consumer.id, **edge_attrs)

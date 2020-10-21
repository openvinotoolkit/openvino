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

import numpy as np

from mo.front.common.layout import get_features_dim, shape_for_layout
from mo.front.tf.graph_utils import create_op_with_const_inputs
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
        for node in graph.get_op_nodes():
            for out_port_idx in node.out_ports():
                mapping = {}
                output_port = node.out_port(out_port_idx)
                for consumer_port in output_port.get_destinations():
                    edge_attrs = consumer_port.get_in_edge_attrs()
                    if 'new_shape' in edge_attrs:
                        if np.array_equal(edge_attrs['new_shape'], output_port.data.get_shape()):
                            continue
                        new_shape = tuple([x for x in edge_attrs['new_shape']])
                        if not new_shape in mapping:
                            mapping.update({new_shape: [consumer_port]})
                        else:
                            mapping[new_shape].append(consumer_port)

                # Insert Reshape layer between data node and consumer
                for shape_key in mapping.keys():
                    new_shape = list(shape_key)
                    reshape_name = node.soft_get('name', node.id) + '/EltwiseReshape'
                    reshape_node = create_op_with_const_inputs(graph, Reshape, {1: new_shape},
                                                                {'name': reshape_name})
                    reshape_node.in_port(0).connect(output_port)

                    # Iterate over consumers and reconnect them to Reshape layer output
                    for consumer_port in mapping[shape_key]:
                        edge_attrs = consumer_port.get_in_edge_attrs()
                        del edge_attrs['new_shape']
                        consumer_port.connect(reshape_node.out_port(0))

                    # Adjust shape and value for Reshape output
                    output_port_value = output_port.data.get_value()
                    if output_port_value is not None:
                        reshape_node.out_port(0).data.set_value(np.reshape(output_port_value, new_shape))
                    else:
                        reshape_node.out_port(0).data.set_shape(new_shape)

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

from extensions.ops.gather import Gather
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.concat import Concat
from mo.ops.const import Const


def node_to_get_shape_value_of_range(shape_node: Node, indices: list):
    """
    The function returns a node that produces values of the specified indices of the input node 'shape_node'

    :param shape_node: the node of 1D output shape to get elements from
    :param indices: the list of element indices to get
    :return: node producing required elements of the node
    """
    graph = shape_node.graph
    indices_node = Const(graph, {'value': int64_array(indices), 'name': shape_node.name + '/Indices'}).create_node()
    gather_node = Gather(graph, {'name': shape_node.name + '/Gather'}).create_node()

    shape_node.out_port(0).connect(gather_node.in_port(0))
    indices_node.out_port(0).connect(gather_node.in_port(1))

    return gather_node


def node_to_get_batch_value(shape_node: Node):
    """
    The function returns a node that produces the batch value which is usually the element of the shape with index 0
    :param shape_node: the node of 1D output shape to get batch from
    :return: the node producing batch value
    """
    return node_to_get_shape_value_of_range(shape_node, [0])


def node_to_get_features_dimension_value(shape_node: Node):
    """
    The function returns a node that produces the feature dimension value
    :param shape_node: the node of 1D output shape to get the feature dimension value from
    :return: the node producing feature dimension value
    """
    layout = shape_node.graph.graph['layout']
    if layout == 'NCHW':
        return node_to_get_shape_value_of_range(shape_node, [1])
    elif layout == 'NHWC':
        return node_to_get_shape_value_of_range(shape_node, [-1])
#        return node_to_get_shape_value_of_range(graph, shape_node, [len(shape_node.in_port(0).get_connection().get_source().data.get_shape()) - 1])
    else:
        assert 'Unsupported layout "{}"'.format(layout)


def node_to_get_spatial_dimensions_value(shape_node: Node):
    """
    The function returns a node that produces the spatial dimension values
    :param shape_node: the node of 1D output shape to get the spatial dimension values from
    :return: the node producing the spatial dimension values
    """
    layout = shape_node.graph.graph['layout']
    shape = shape_node.in_port(0).get_connection().get_source().data.get_shape()
    assert shape is not None, 'The shape must be inferred before running this function'

    if layout == 'NCHW':
        return node_to_get_shape_value_of_range(shape_node, list(range(2, len(shape))))
    elif layout == 'NHWC':
        return node_to_get_shape_value_of_range(shape_node, list(range(1, len(shape) - 1)))
    else:
        assert 'Unsupported layout "{}"'.format(layout)


def new_shape_node_from_shape_nodes(input_shape_nodes: list):
    """
    The function returns a node producing 1D tensor with concatenated shapes produced by nodes from "input_shape_nodes"
    :param input_shape_nodes: list of nodes producing 1D tensors
    :return: the node producing concatenated values of nodes from the "input_shape_nodes"
    """
    assert len(input_shape_nodes) > 0, 'The list of input shape nodes should be non-empty'
    new_shape_node = Concat(input_shape_nodes[0].graph, {'axis': 0}).create_node()

    for ind, input_node in enumerate(input_shape_nodes):
        new_shape_node.add_input_port(ind)
        new_shape_node.in_port(ind).connect(input_node.out_port(0))
    return new_shape_node

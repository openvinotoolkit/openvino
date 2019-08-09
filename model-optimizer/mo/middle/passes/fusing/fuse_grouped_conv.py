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

import logging as log
from collections import deque

import networkx as nx
import numpy as np

from mo.front.extractor import add_attrs_props
from mo.graph.graph import Node, Graph
from mo.middle.passes.eliminate import graph_clean_up
from mo.middle.passes.fusing.helpers import get_next_operation, get_tensor_id


# TODO: unit tests
def concat_convolutions(graph: Graph, start_node: Node, last_node: Node):
    """
    This function converts group of convolutions into one
    """

    # Check that concatenation makes in the same order
    conv_nodes = get_next_operation(start_node)
    assert len(conv_nodes) == len(last_node.in_nodes())
    gconv = conv_nodes[0]

    for id in range(len(conv_nodes)):
        conv = conv_nodes[id]
        if conv.out_node().id != last_node.in_node(id).id:
            return False
        # Check that all convolutions have same weights shapes
        if not np.array_equal(conv.in_node(1).shape, gconv.in_node(1).shape):
            log.debug('Grouped convolutions fusion : convolutions have different weights shape')
            return False

    # Check that split and concat dims are valid
    channel_dim = gconv.channel_dims[0]
    if channel_dim != start_node.axis or channel_dim != last_node.axis:
        log.debug('Grouped convolutions fusion : split or concat has wierd axis!')
        return False

    # Check that all convolutions has the same parameters
    conv_attrs = ['pad', 'stride']
    for attr in conv_attrs:
        for id in range(len(conv_nodes)):
            conv = conv_nodes[id]
            if not np.array_equal(gconv[attr], conv[attr]):
                log.debug('Grouped convolutions fusion : attrs {} doesn\'t match'.format(attr))
                return False

    # Check that all Convolutions has biases (if exists)
    has_biases = False
    for id in range(len(conv_nodes)):
        conv = conv_nodes[id]
        if len(conv.in_nodes()) == 3:
            if not has_biases:
                has_biases = True
        elif has_biases:
            return False  # All convolution mast have biases

    # Check that all biases have same shape
    if has_biases:
        for id in range(len(conv_nodes)):
            conv = conv_nodes[id]
            if conv.in_node(2).shape != gconv.in_node(2).shape:
                log.debug('Group convolutions fusion : convolutions have different biases shape {} and {}'.format(
                    conv.in_node(2).shape, gconv.in_node(2).shape))
                return False

    graph.remove_edge(gconv.in_node(0).id, gconv.id)
    graph.remove_edge(gconv.id, gconv.out_node().id)

    input = start_node.in_node(start_node.input_port)
    output = last_node.out_node()

    # Removing edges from data nodes to Split and Concat
    graph.remove_edge(input.id, start_node.id)
    graph.remove_edge(last_node.id, output.id)

    # Add edges to grouped convolution
    graph.add_edges_from([
        (input.id, gconv.id, {'in': 0}),
        (gconv.id, output.id, {'out': 0})
    ])

    # Concatenation of convolutions
    weights_node = gconv.in_node(1)
    bias_node = gconv.in_node(2) if has_biases else None

    weights_value = np.array(weights_node.value)
    bias_value = np.array(bias_node.value) if has_biases else None

    feature_dim = 3 if graph.graph['layout'] == 'NHWC' else 1

    for conv in conv_nodes[1:]:
        weights_value = np.concatenate((weights_value, conv.in_node(1).value), axis=feature_dim)
        if has_biases:
            bias_value = np.concatenate((bias_value, conv.in_node(2).value), axis=-1)  # Not validated

    weights_node.value = np.array(weights_value)
    weights_node.shape = np.array(weights_value.shape)

    if has_biases:
        bias_node.value = np.array(bias_value)
        bias_node.shape = np.array(bias_value.shape)

    log.debug('Start node : {} Last node : {}  Nodes inside : {}'.format(start_node.id, last_node.id,
                                                                         len(start_node.out_nodes())))
    log.debug('Output shape : {}'.format(weights_value.shape))

    gconv.group = len(conv_nodes)
    gconv.output = weights_node.shape[feature_dim]
    gconv.output_shape[feature_dim] = weights_node.shape[feature_dim]

    return True


# TODO: unit tests
def grouped_convolutions_fusing(graph: Graph):
    while True:
        is_fused = False
        graph_clean_up(graph, ['TFCustomSubgraphCall', 'ShapeOf', 'Shape'])
        for node in graph.pseudo_topological_sort():
            if node.kind == 'op' and len(node.out_nodes()) > 1:
                if node.soft_get('can_be_fused') == False:
                    continue

                is_valid_convolutions = True
                last_layer = None

                next_nodes = get_next_operation(node)
                # Check that all operation after this one are Convolutions
                # and all convolutions has same output
                if len(next_nodes) > 1 and all(_node.soft_get('type') in ['Convolution', 'Deconvolution'] for _node in next_nodes):
                    for conv in next_nodes:
                        conv_outputs = get_next_operation(conv)
                        if conv.soft_get('can_be_fused') == False:
                            is_valid_convolutions = False
                        if len(conv_outputs) != 1:
                            is_valid_convolutions = False
                        if last_layer is None:
                            last_layer = conv_outputs[0].id
                        elif conv_outputs[0].id != last_layer:
                            is_valid_convolutions = False

                    if is_valid_convolutions:
                        is_fused = concat_convolutions(graph, node, Node(graph, last_layer))
                        if is_fused:
                            break

        if not is_fused:
            break

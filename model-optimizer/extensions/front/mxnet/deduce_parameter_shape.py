# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import List

import networkx as nx
import numpy as np
import logging as log
from mo.front.common.partial_infer.utils import dynamic_dimension, shape_array, unmask_shape
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph
from mo.graph.graph import Node


class DeduceParameterShape(FrontReplacementSubgraph):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'mxnet' and graph.graph['user_shapes'] is None]

    def find_and_replace_pattern(self, graph: Graph):
        convolutions = graph.get_op_nodes(op='Convolution')
        parameter_nodes = graph.get_op_nodes(op='Parameter')
        for parameter in parameter_nodes:
            conv = find_nearest_conv(graph, parameter, convolutions)
            if conv.in_port(1).disconnected() or not conv.in_node(1).has_valid('value'):
                continue

            conv_shape = conv.in_node(1).value.shape
            parameter_rank = len(conv_shape)
            num_space_dimensions = parameter_rank - 2
            num_channels = conv_shape[1]
            parameter_shape = [dynamic_dimension, num_channels, *([dynamic_dimension] * num_space_dimensions)]
            parameter['shape'] = shape_array(parameter_shape)
            log.error('Deduced input_shape={} for input \'{}\' from Convolution node \'{}\' weights'.format(
                unmask_shape(shape_array(parameter_shape)), parameter.soft_get('name', parameter.id),
                conv.soft_get('name', conv.id)),
                extra={'is_warning': True})


def find_nearest_conv(graph: Graph, start_node: Node, end_nodes: List[Node]) -> Node:
    found_shortest_path = None
    shortest_path_len = np.iinfo(np.int64).max
    for conv_node in end_nodes:
        if nx.has_path(graph, start_node.id, conv_node.id):
            path = nx.shortest_path(graph, start_node.id, conv_node.id)  # pylint: disable-msg=E1121
            if len(path) == 2:  # if input goes directly into convolution
                found_shortest_path = path
                break
            if len(path) < shortest_path_len:
                found_shortest_path = path
                shortest_path_len = len(path)
    return Node(graph, found_shortest_path[-1])

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import List

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.utils import dynamic_dimension, shape_array
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
            if conv.in_port(1).disconnected() or not hasattr(conv.in_node(1), 'value'):
                continue

            conv_shape = conv.in_node(1).value.shape
            parameter_rank = len(conv_shape)
            num_space_dimensions = parameter_rank - 2
            num_channels = conv_shape[1]
            parameter_shape = [dynamic_dimension, num_channels, *([dynamic_dimension] * num_space_dimensions)]
            parameter['shape'] = shape_array(parameter_shape)


def find_nearest_conv(graph: Graph, parameter_node: Node, convolution_nodes: List[Node]) -> Node:
    shortest_path = None
    shortest_path_len = np.iinfo(np.int64).max
    for conv_node in convolution_nodes:
        if nx.has_path(graph, parameter_node.id, conv_node.id):
            path = nx.shortest_path(graph, parameter_node.id, conv_node.id)
            if len(path) == 2:  # if input goes directly into convolution
                shortest_path = path
                break
            if len(path) < shortest_path_len:
                shortest_path = path
                shortest_path_len = len(path)
    return Node(graph, shortest_path[-1])

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log
from typing import List, Iterable

import networkx as nx
import numpy as np

from mo.front.common.partial_infer.utils import dynamic_dimension, shape_array, unmask_shape
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph, Node


class DeduceParameterShape(FrontReplacementSubgraph):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].input is None]

    def run_before(self):
        from extensions.front.output_cut import OutputCut
        from extensions.front.input_cut import InputCut
        return [OutputCut, InputCut]

    def run_after(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        nodes_with_defined_ranks = get_nodes_with_defined_ranks(graph, deduce_pshape_map.keys())

        parameter_nodes = graph.get_op_nodes(op='Parameter')
        for parameter in parameter_nodes:
            nearest_node = find_nearest_node_with_pshape(graph, parameter, nodes_with_defined_ranks)
            if nearest_node is None or len(nearest_node.in_nodes()) < 2 or not nearest_node.in_node(1).has_valid('value'):
                continue

            deduced_pshape = deduce_pshape_map[nearest_node.type](graph, nearest_node)
            if deduced_pshape is not None:
                parameter['shape'] = deduced_pshape

                log.error('Deduced input_shape={} for input \'{}\' from Convolution node \'{}\' weights'.format(
                    unmask_shape(parameter['shape']), parameter.soft_get('name', parameter.id),
                    nearest_node.soft_get('name', nearest_node.id)),
                    extra={'is_warning': True})


def get_nodes_with_defined_ranks(graph: Graph, ops: Iterable) -> List[Node]:
    res = []
    for op in ops:
        res.extend(graph.get_op_nodes(type=op))
    return res


def find_nearest_node_with_pshape(graph: Graph, start_node: Node, end_nodes: List[Node]) -> Node:
    """

    :param graph:
    :param start_node: Parameter node from which search is begun
    :param end_nodes: end nodes from which partial shape can be deduced: Convolution, Pooling, Transpose, etc.
    :return: the node that is nearest to the Parameter and for which we can deduce partial shape
    """
    shortest_path = None
    shortest_path_len = np.iinfo(np.int64).max
    for end_node in end_nodes:
        if nx.has_path(graph, start_node.id, end_node.id):
            path = nx.shortest_path(graph, start_node.id, end_node.id)  # pylint: disable-msg=E1121
            if len(path) == 2:  # if input goes directly into convolution
                shortest_path = path
                break
            if len(path) < shortest_path_len:
                shortest_path = path
                shortest_path_len = len(path)
    if shortest_path is not None:
        return Node(graph, shortest_path[-1])
    else:
        return None


def get_convolution_input_pshape(graph: Graph, node: Node):
    conv_shape = node.in_node(1).value.shape
    parameter_rank = len(conv_shape)
    num_space_dimensions = parameter_rank - 2

    if graph.graph['fw'] in ['onnx', 'mxnet']:
        num_channels = conv_shape[1]
    elif graph.graph['fw'] == 'tf' and node.type == 'Convolution':
        num_channels = conv_shape[num_space_dimensions]
    elif graph.graph['fw'] == 'tf' and node.type == 'Deconvolution':
        num_channels = conv_shape[num_space_dimensions + 1]
    else:
        return None

    res = shape_array([dynamic_dimension, num_channels, *([dynamic_dimension] * num_space_dimensions)])
    if graph.graph['layout'] == 'NHWC' and parameter_rank > 3:
        new_nhwc_order = [0, *list(range(2, parameter_rank)), 1]
        res = res[new_nhwc_order]
    return res


def get_pooling_input_pshape(graph: Graph, node: Node):
    window = node.soft_get('window', None)
    if window is None and len(node.in_nodes()) > 1 and node.in_node(1).value is not None:
        window = node.in_node(1).value
    if window is not None:
        return [dynamic_dimension] * len(window)

    return None


deduce_pshape_map = {
    # will be uncommented when corresponding logic for deducing will be needed
    # 'StridedSlice': None,
    # 'Reshape': None,
    # 'Transpose': None,
    'Convolution': get_convolution_input_pshape,
    'Deconvolution': get_convolution_input_pshape,
    'Pooling': get_pooling_input_pshape,
    'PoolingV2': get_pooling_input_pshape,
}

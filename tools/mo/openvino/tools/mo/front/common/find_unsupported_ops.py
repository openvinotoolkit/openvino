# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.graph.graph import Node, Graph


def find_unsupported_ops(graph: Graph):
    """
    The function returns list of node name those are not supported. Currently nodes that product non FP32 data tensors
    or has undefined 'type' attribute are considered unsupported.
    :param graph: current graph with operations. Data nodes are not yet added.
    :return: the list of node names which are not supported
    """
    unsupported = list()
    for node_name in graph.nodes():
        node = Node(graph, node_name)
        # op node that produce non FP32 data or has no type are considered unsupported
        if node.kind == 'op':
            if node.has_valid('type') or (node.has_valid('op') and node.op == 'Result'):
                for out_data_node in node.out_nodes().values():
                    if out_data_node.has_valid('data_type') and out_data_node.data_type != np.float32:
                        log.info('Node "{}" produces output as non FP32. Consider it unsupported'.format(node_name))
                        unsupported.append(node.id)
            else:
                log.info('Node "{}" does not have type. Consider it unsupported'.format(node_name))
                unsupported.append(node.id)
    return unsupported


# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re

from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.MarkSubgraphsWithCorrectLayout import MarkSubGraphsWithCorrectLayout
from openvino.tools.mo.middle.passes.fusing.helpers import get_value_id


def _check_lin_op(node: Node, layout: str):
    lin_ops = ['Mul', 'Add']
    if node.soft_get('op') in lin_ops:
        weights_id = get_value_id(node)
        if weights_id is None:
            node.graph.node[node.id]['can_be_fused'] = False
            log.info('[ FUSING ] Node {} wasn\'t marked as fusable (no weights, probably this is element-wise operation'
                     ' that is not fusable)'.format(node.id))
            return

        node.graph.node[node.id]['can_be_fused'] = True
        log.info('[ FUSING ] Node {} marked as fusable'.format(node.id))


def mark_unfused_nodes(graph: Graph, regex_masks: str):
    regex_masks = [] if not regex_masks else regex_masks.split(',')
    nodes = graph.get_op_nodes()
    for node in nodes:
        if node.has_valid('can_be_fused'):
            continue
        disabled = False
        for mask in regex_masks:
            res = re.findall(mask, node.name)
            if res and len(res):
                graph.node[node.id]['can_be_fused'] = False
                log.info('[ FUSING ] Node {} wasn\'t marked as fusable (user decision {})'.format(node.id,mask))
                disabled = True
        if not disabled:
            _check_lin_op(node, graph.graph['layout'])


def mark_shape_of_sugraph_as_unfusable(graph: Graph):
    def condition_to_continue(node: Node):
        for port in node.out_ports().values():
            if port.data.get_value() is None:
                return False
        return True

    starting_nodes = graph.get_op_nodes(op='ShapeOf')
    shapeof_subgraph_nodes = MarkSubGraphsWithCorrectLayout.bfs(starting_nodes, set(), condition_to_continue)

    for node in shapeof_subgraph_nodes:
        node['can_be_fused'] = False

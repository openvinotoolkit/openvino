import logging as log
import networkx as nx
import numpy as np
import re

from mo.graph.graph import get_graph_ops, Node
from mo.middle.passes.fusing.helpers import get_value_id


def _check_lin_op(node: Node, layout: str):
    lin_ops = ['Mul', 'Add']
    if node.soft_get('op') in lin_ops:
        weights_id = get_value_id(node)
        if weights_id is None:
            node.graph.node[node.id]['can_be_fused'] = False
            log.info('[ FUSING ] Node {} wasn\'t marked as fusable (no weights, probably this is Eltwise that is not fusable)'.format(node.id))
            return

        weights_shape = np.array(node.in_node(weights_id).shape)
        node.graph.node[node.id]['can_be_fused'] = True
        log.info('[ FUSING ] Node {} marked as fusable'.format(node.id))


def mark_unfused_nodes(graph: nx.MultiDiGraph, regex_masks: str):
    regex_masks = [] if not regex_masks else regex_masks.split(',')
    nodes = get_graph_ops(graph)
    for node in nodes:
        disabled = False
        for mask in regex_masks:
            res = re.findall(mask, node.name)
            if res and len(res):
                graph.node[node.id]['can_be_fused'] = False
                log.info('[ FUSING ] Node {} wasn\'t marked as fusable (user decision {})'.format(node.id,mask))
                disabled = True
        if not disabled:
            _check_lin_op(node, graph.graph['layout'])



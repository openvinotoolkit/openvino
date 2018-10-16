import logging as log
from collections import deque

import networkx as nx
import numpy as np

from mo.front.extractor import add_attrs_props
from mo.graph.graph import Node, unique_id
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.graph import pseudo_topological_sort
from mo.ops.lin_op import Mul, Add
from mo.ops.op import Op
from mo.graph.graph import dump_graph_for_graphviz


def get_value_id(node: Node):
    assert node.has_valid('op') and (node.op == 'Mul' or node.op == 'Add')
    value_id = None
    for port, in_node in node.in_nodes().items():
        if in_node.value is not None:
            if value_id:
                return None
            value_id = port
    return value_id


def get_tensor_id(node: Node):
    assert node.has_valid('op') and (node.op == 'Mul' or node.op == 'Add')
    tensor_id = None
    for port, in_node in node.in_nodes().items():
        if in_node.value is None:
            if tensor_id:
                return None
            tensor_id = port
    return tensor_id


def common_bfs(start_node: Node, allowed_ops: list, op_name: list, is_backward: bool = True, allowed_all: bool = False):
    """
    The purpose of this algorithm is to find layers with 'op_name' located in given direction.
    In case of branching algorithm goes into each branch, but if it can't find layer in one of them it returns
    empty list.

    :param start_node: Start node for BFS algorithm
    :param allowed_ops: List of operations that we can jump over
    :param op_name: The list with names of operations for searching
    :param is_backward: The direction of BFS algorithm
    :param allowed_all: Bool flag meaning we can jump over all operations
    """
    ret = []
    q = deque([start_node])
    used = []
    while len(q) != 0:
        node = q.popleft()
        if node.id in used:
            log.debug("[BFS:ERROR] Graph contains cycle! BFS starts from {} node".format(start_node.id))
            return []
        used.append(node.id)
        in_nodes_size = len(node.in_nodes()) if is_backward else len(node.out_nodes())
        for id in range(in_nodes_size):  # in_nodes() can return either list or dict
            pnode = node.in_node(id) if is_backward else node.out_node(id)
            if pnode.has_valid('type'):
                if pnode.type in op_name:
                    if pnode.id not in ret:
                        ret.append(pnode.id)
                elif allowed_all or pnode.op in allowed_ops:
                    q.append(pnode)
                else:
                    return []
            elif pnode.kind == 'data' and pnode.value is None:
                # If we go backward we don't use data node that have more than one consumer
                if not is_backward or (is_backward and len(pnode.out_nodes()) == 1):
                    q.append(pnode)
    return [Node(start_node.graph, x) for x in ret]


def forward_bfs(start_node: Node, allowed_ops: list, op_name: list, allowed_all: bool = False):
    return common_bfs(start_node, allowed_ops, op_name, False, allowed_all=allowed_all)


def backward_bfs(start_node: Node, allowed_ops: list, op_name: list, allowed_all: bool = False):
    return common_bfs(start_node, allowed_ops, op_name, allowed_all=allowed_all)


def get_next_operation(node: Node):
    """
    This function returns next op node, so node should be an operation
    """
    assert node.kind == 'op'

    out_nodes = node.out_nodes()
    res = []
    for port, out_node in out_nodes.items():
        op_nodes = out_node.out_nodes()
        for op_node in op_nodes:
            if op_node.id not in [n.id for n in res]:
                res.append(op_node)
    return res

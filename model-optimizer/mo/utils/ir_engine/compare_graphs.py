# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from numbers import Number

import numpy as np

from mo.graph.graph import Graph, Node


def compare_graphs(graph: Graph, graph_ref: Graph, last_node: str, last_node_ref=None, check_op_attrs=False):
    from mo.utils.ir_engine.ir_engine import IREngine
    stderr = []
    if last_node_ref is None:
        last_node_ref = last_node

    if 'statistics' in graph.graph and 'statistics' in graph_ref.graph:
        assert graph.graph['statistics'] == graph_ref.graph['statistics'], "int8 statistics comparison failed"

    q = deque([last_node])
    q_ref = deque([last_node_ref])

    checked_nodes = []
    checked_nodes_ref = []

    while len(q_ref) != 0:
        if len(q) == 0:
            stderr.append('Graphs have different number of nodes')
            return (False, stderr)
        node = Node(graph, q.popleft())
        node_ref = Node(graph_ref, q_ref.popleft())

        checked_nodes.append(node.id)
        checked_nodes_ref.append(node_ref.id)

        # Check that nodes has same amount of output nodes
        if len(node_ref.out_nodes()) != len(node.out_nodes()):
            stderr.append('Current node "{}" and reference node "{}" have different amount of output nodes: {} vs {}'.\
                          format(node.id, node_ref.id, len(node.out_nodes()), len(node_ref.out_nodes())))
            continue

        # Check that nodes has same amount of input nodes
        if len(node_ref.in_nodes()) != len(node.in_nodes()):
            stderr.append('Current node "{}" and reference node "{}" have different amount of input nodes: {} vs {}'.\
                          format(node.id, node_ref.id, len(node.in_nodes()), len(node_ref.in_nodes())))
            continue

        # Check that nodes has same 'kind'
        if node_ref.kind != node.kind:
            stderr.append('Current node "{}" and reference node "{}" have different kind parameter'.\
                          format(node.id, node_ref.id))
            return (False, stderr)

        # Check can_be_fused attr
        if node_ref.has_valid('can_be_fused'):
            if node_ref.soft_get('can_be_fused') != node.soft_get('can_be_fused'):
                stderr.append('Current node "{}" and reference node "{}" have different "can_be_fused" parameter ' \
                              '{} and {}'.format(node.id, node_ref.id, node.soft_get('can_be_fused'),
                                                 node_ref.soft_get('can_be_fused')))

        if node_ref.kind == 'op':
            # Check that nodes has same operation
            if check_op_attrs:
                cur_node_type = node.type if node.has_valid("type") else None
                ref_node_type = node_ref.type if node_ref.has_valid("type") else None
                for attr in graph_ref.node[node_ref.id]:
                    if graph_ref.node[node_ref.id][attr] is None or attr in ['name', 'id', '_in_ports', '_out_ports',
                                                                             'infer', 'IE', 'biases', 'weights', 'custom', 'offset']:
                        continue
                    if attr not in graph.node[node.id]:
                        stderr.append('Current node "{}" with type {} has missing attribute {}'.format(node.id, cur_node_type, attr))
                        continue

                    if attr == 'value':
                        if not values_are_equal(node.value, node_ref.value):
                            stderr.append('Current node "{}" with type {} and reference node "{}" with type have different values '
                                          '\n{} \nand \n{}'.format(node.id, cur_node_type, node_ref.id, ref_node_type, node.value, node_ref.value))
                        continue

                    def err_format_string():
                        return 'Current node "{}" with type {} and reference node "{}" with type {} have different attr "{}" : ' \
                                '{} and {}'.format(node.id, cur_node_type, node_ref.id, ref_node_type, attr,
                                                   graph.node[node.id][attr],
                                                   graph_ref.node[node_ref.id][attr])

                    if type(graph_ref.node[node_ref.id][attr]) in [np.ndarray, list]:
                        if not np.array_equal(graph.node[node.id][attr], graph_ref.node[node_ref.id][attr]):
                            stderr.append(err_format_string())
                    elif isinstance(graph.node[node.id][attr], Number):
                        eps = 5e-2 if node.has('precision') and node['precision'] == 'FP16' else 1e-4
                        if abs(graph.node[node.id][attr] - graph_ref.node[node_ref.id][attr]) > eps:
                            stderr.append(err_format_string())
                    elif isinstance(graph.node[node.id][attr], IREngine):
                        resp, err_log = graph.node[node.id][attr].compare(graph_ref.node[node_ref.id][attr])
                        if not resp:
                            stderr.extend(err_log)
                    elif isinstance(graph.node[node.id][attr], np.ndarray):
                        if not np.array_equal(graph.node[node.id][attr], graph_ref.node[node_ref.id][attr]):
                            stderr.append(err_format_string())
                    elif graph.node[node.id][attr] != graph_ref.node[node_ref.id][attr]:
                        stderr.append(err_format_string())
        else:
            if node_ref.has_valid('shape') and not node.has_valid('shape'):
                stderr.append('{} has None shape'.format(node.id))
            if node_ref.has_valid('value') and not node.has_valid('value'):
                stderr.append('{} has None value'.format(node.id))

            # Check that nodes has same shape and value
            if node_ref.has_valid('shape') and node_ref.shape is not None and not np.array_equal(node_ref.shape,
                                                                                                 node.shape):
                stderr.append('Current node "{}" and reference node "{}" have different shapes {} and {}'.\
                              format(node.id, node_ref.id, node.shape, node_ref.shape))
                continue

            if node_ref.has_valid('value') and node_ref.value is not None and \
                    not values_are_equal(node.value, node_ref.value):
                stderr.append('Current node "{}" and reference node "{}" have different values \n{} \nand \n{}'.\
                              format(node.id, node_ref.id, node.value, node_ref.value))
        ports = sorted(node.in_nodes().keys()) if node.kind == 'op' else None
        in_nodes = [node.in_node(k) for k in ports] if node.kind == 'op' else node.in_nodes()
        for in_node in in_nodes:
            if in_node.id not in checked_nodes and in_node.id not in q:
                q.append(in_node.id)

        ports_ref = sorted(node_ref.in_nodes().keys()) if node_ref.kind == 'op' else None
        if ports != ports_ref:
            stderr.append('Current node "{}" and reference node "{}" have different ports'.format(node.id, node_ref.id))
            return (False, stderr)

        in_nodes = [node_ref.in_node(k) for k in ports] if node_ref.kind == 'op' else node_ref.in_nodes()
        for in_node in in_nodes:
            if in_node.id not in checked_nodes_ref and in_node.id not in q_ref:
                q_ref.append(in_node.id)

        if node.kind == 'op':
            out_nodes = sorted_by_name([Node(graph, v) for v, _ in node.get_outputs()])
        else:
            out_nodes = sorted_by_name(node.out_nodes())
        for out_node in out_nodes:
            if out_node.id not in checked_nodes and out_node.id not in q:
                q.append(out_node.id)

        if node_ref.kind == 'op':
            out_nodes = sorted_by_name([Node(graph_ref, v) for v, _ in node_ref.get_outputs()])
        else:
            out_nodes = sorted_by_name(node_ref.out_nodes())
        for out_node in out_nodes:
            if out_node.id not in checked_nodes_ref and out_node.id not in q_ref:
                q_ref.append(out_node.id)

    return (False, stderr) if stderr else (True, [])


def sorted_by_name(nodes_list):
    return sorted(nodes_list, key=lambda x: x.soft_get('name', x.id))


def values_are_equal(value, value_ref):
    dtype = np.asarray(value).dtype
    if dtype == 'uint8':
        eps = 0
    elif dtype == 'float16':
        eps = 5e-2
    else:
        eps = 1e-4
    return np.allclose(value_ref, value, rtol=eps, atol=eps)


"""
 Copyright (c) 2018 Intel Corporation

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
from mo.graph.graph import Node, unique_id
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.graph import pseudo_topological_sort
from mo.ops.lin_op import Mul, Add
from mo.middle.passes.eliminate import merge_data_nodes
from mo.ops.op import Op
from mo.graph.graph import dump_graph_for_graphviz
from mo.middle.passes.fusing.helpers import backward_bfs, forward_bfs, get_tensor_id, get_value_id


def _fuse_linear_sequence(graph: nx.MultiDiGraph, start_node: Node):
    """
    This function finds the sequence of Mul/Add operations and replaces this sequence with two ops (Mul->Add).
    :param graph:
    :param start_node: The first operation of the sequence
    """
    fnodes = [start_node]
    while True:
        node = fnodes[-1]
        data_node = node.out_node()
        if (len(data_node.out_nodes()) != 1):
            break
        if (data_node.out_node().op in ['Mul', 'Add']) and get_value_id(data_node.out_node()) is not None and data_node.out_node().soft_get('can_be_fused') == True:
            fnodes.append(data_node.out_node())
        else:
            break

    if len(fnodes) == 1 or (len(fnodes) == 2 and fnodes[0].op == 'Mul' and fnodes[1].op == 'Add'):
        return False

    input_shape = start_node.in_node(get_tensor_id(start_node)).shape

    init_dims_cnt = len(input_shape) - 2 if graph.graph['layout'] == 'NCHW' else 1

    mul = np.ones([1 for x in range(init_dims_cnt)])
    add = np.zeros([1 for x in range(init_dims_cnt)])

    first_mul_name = None
    first_add_name = None

    for idx in range(len(fnodes)):
        node = fnodes[idx]
        const_node = get_value_id(node)
        if node.op == 'Mul':
            if first_mul_name is None:
                first_mul_name = node.name
            mul = mul * node.in_node(const_node).value
            add = add * node.in_node(const_node).value
        elif node.op == 'Add':
            if first_add_name is None:
                first_add_name = node.name
            add = add + node.in_node(const_node).value

    # If mul is scalar we broadcast it to biases shape
    if mul.shape != add.shape and len(mul.shape) == 1 and mul.shape[0] == 1:
        mul = np.array([mul[0] for x in range(add.shape[0])])

    assert (np.array_equal(fnodes[0].in_node(get_tensor_id(fnodes[0])).shape, fnodes[-1].out_node().shape))

    mul_node = Mul(graph, dict(name=first_mul_name + '/Fused_Mul_' if first_mul_name is not None else ''))
    add_node = Add(graph, dict(name=first_add_name + '/Fused_Add_' if first_add_name is not None else ''))

    in_node = fnodes[0].in_node(get_tensor_id(fnodes[0]))
    out_node = fnodes[-1].out_node()

    graph.remove_edge(in_node.id, fnodes[0].id)
    graph.remove_edge(fnodes[-1].id, out_node.id)

    # Remove deleted subgraph
    for node in fnodes:
        for tmp_node in node.in_nodes().values():
            # Remove node only if it has one consumer (for case with shared weights)
            if len(tmp_node.out_nodes()) == 1:
                graph.remove_node(tmp_node.id)
        for tmp_node in node.out_nodes().values():
            graph.remove_node(tmp_node.id)
        graph.remove_node(node.id)

    """
    Four cases considered below:
        1. Mul and Add have valid values (mul value != 1 and add value != 0)
        2. Only Mul has valid values, so we add only Mul node
        3. Only Add has valid values, so we add only Add node
        4. When Mul and Add has not valid values we just merge two data nodes
    """
    if any([x != 0 for x in np.nditer(add)]) and any([x != 1 for x in np.nditer(mul)]):
        data_mul = Op.create_input_data_node(graph, "data_mul_", np.array(mul))
        data_add = Op.create_input_data_node(graph, "data_add_", np.array(add))
        add_node.create_node_with_data(inputs=[mul_node.create_node_with_data([in_node, data_mul]), data_add],
                                       data_nodes=out_node)
    elif any([x != 1 for x in np.nditer(mul)]):
        data_mul = Op.create_input_data_node(graph, "data_mul_", np.array(mul))
        mul_node.create_node_with_data(inputs=[in_node, data_mul], data_nodes=out_node)
    elif any([x != 0 for x in np.nditer(add)]):
        data_add = Op.create_input_data_node(graph, "data_add_", np.array(add))
        add_node.create_node_with_data(inputs=[in_node, data_add], data_nodes=out_node)
    else:
        merge_data_nodes(graph,out_node, in_node)
        graph.remove_node(in_node.id)

    log.debug('Fused {} operations'.format(len(fnodes)))
    return True


def fuse_mul_add_sequence(graph: nx.MultiDiGraph):
    """
    This function finds first valid Mul/Add node and pass it to fuse_linear_sequence where full sequence will be found
    """
    while True:
        is_fused = False
        for idx in list(pseudo_topological_sort(graph)):
            if idx in graph:
                node = Node(graph, idx)
                if node.soft_get('op') in ['Mul','Add'] and get_value_id(node) is not None and node.soft_get('can_be_fused') == True:
                    is_fused |= _fuse_linear_sequence(graph, node)
        if not is_fused:
            break
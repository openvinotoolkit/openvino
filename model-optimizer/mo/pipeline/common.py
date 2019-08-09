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
import os
from operator import itemgetter

import networkx as nx

from mo.back.ie_ir_ver_2.emitter import port_renumber, serialize_constants, generate_ie_ir, serialize_mean_image
from mo.graph.graph import Node, Graph
from mo.middle.passes import tensor_names, convert_data_type
from mo.utils.error import Error


def determined_sort(outputs: list):
    op_order = []
    data_order = []
    stack = list(outputs)
    visited = set()
    while len(stack) != 0:
        node = stack[0]
        node_id = node.id
        stack.pop(0)
        visited.add(node_id)
        has_child = False
        in_names = [n.id if isinstance(node.in_nodes(), list) else node.in_node(n).id for n in node.in_nodes()]
        for in_node_name in in_names:
            if in_node_name not in visited:
                stack.insert(0, node)
                stack.insert(0, Node(node.graph, in_node_name))
                has_child = True
                break
        if not has_child:
            if node.kind == 'op':
                op_order.append(node_id)
            if node.kind == 'data':
                data_order.append(node_id)
    return op_order, data_order


def get_fw_tensor_debug_info(node: Node):
    while not node.has_valid('fw_tensor_debug_info') and not node.has_valid('output_sort_order') \
            and len(node.in_nodes()):
        try:
            node = node.in_node()
        except Exception as e:
            log.warning('Was not able to determine tensor debug info for node {}'.format(node.name))
            return "dummy_node_name"
    if node.has_valid('output_sort_order'):
        return node.soft_get('output_sort_order')
    return node.soft_get('fw_tensor_debug_info')


def get_sorted_outputs(graph: Graph):
    outputs = []
    outputs_for_sort = {}
    for node in graph.nodes():
        if len(graph.out_edges(node)) == 0:
            outputs.append(Node(graph, node))
    if len(outputs) == 1:
        return outputs
    for node in outputs:
        debug_info = get_fw_tensor_debug_info(node)
        if isinstance(debug_info, str):
            outputs_for_sort[node.id] = debug_info
        elif isinstance(debug_info, list):
            outputs_for_sort[node.id] = debug_info[0][0] + '_' + str(debug_info[0][1])
        else:
            raise Error('Unsupported type of the variable with debug information used to sort output nodes')
    if len(outputs_for_sort) != len(set(outputs_for_sort.values())):
        log.warning('There are at least two output nodes with the same key used to sort the outputs. This means that '
                    'IRs with different order of nodes may be generated between Model Optimizer runs. The dictionary '
                    'with outputs is: {}'.format(outputs_for_sort))
    return [Node(graph, key) for key, value in sorted(outputs_for_sort.items(), key=itemgetter(1))]


def collect_sub_graphs(graph: Graph):
    ''' Go over all nodes and sub_graphs in the graph recursively; returns all found sub-graphs. '''
    result = []
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('sub_graphs'):
            for sub_graph in node.sub_graphs:
                result.append(node[sub_graph])
                result += collect_sub_graphs(node[sub_graph])
    return result


def relabel_nodes_inplace_safe(graph: Graph, new_labels: dict):
    ''' Safely relabels graph in-place without graph copy.
        
        Safity in this place means that it is guarantied that
        there won't be collisions during relabiling process.
    '''
    # Relabel nodes in two stages
    intermediate_map = {node: graph.unique_id('__relabel__{}__'.format(str(i))) for i, node in enumerate(graph.nodes())}
    final_map = {dst: new_labels[src] for src, dst in intermediate_map.items()}
    assert len(set(intermediate_map.keys()).intersection(set(intermediate_map.values()))) == 0
    assert len(set(final_map.keys()).intersection(set(final_map.values()))) == 0
    nx.relabel_nodes(graph, intermediate_map, copy=False)
    nx.relabel_nodes(graph, final_map, copy=False)


def prepare_emit_ir(graph: Graph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = [], meta_info: dict = dict()):
    graph.strict_mode = False
    for sub_graph in [graph] + collect_sub_graphs(graph):
        op_order, data_order = determined_sort(get_sorted_outputs(sub_graph))
        mapping = {v: u for u, v in enumerate(op_order)}
        mapping.update({v: u for u, v in enumerate(data_order, start=len(sub_graph))})
        relabel_nodes_inplace_safe(sub_graph, mapping)
        port_renumber(sub_graph)
        convert_data_type.convert(sub_graph, data_type)

    tensor_names.propagate_op_name_to_tensor(graph)

    bin_file = os.path.join(output_dir, '{}.bin'.format(output_model_name))
    serialize_constants(graph, bin_file)

    mean_offset = None
    mean_size = None
    if mean_data:
        mean_offset, mean_size = serialize_mean_image(bin_file, mean_data=mean_data)

    generate_ie_ir(graph=graph,
                   file_name=os.path.join(output_dir, '{}.xml'.format(output_model_name)),
                   input_names=input_names,
                   mean_offset=mean_offset,
                   mean_size=mean_size,
                   meta_info=meta_info)
    tensor_names.output_tensor_names_map(graph, os.path.join(output_dir, '{}.mapping'.format(output_model_name)))

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

import os
from operator import itemgetter

import networkx as nx

from mo.back.ie_ir_ver_2.emitter import port_renumber, serialize_constants, generate_ie_ir, serialize_mean_image, \
    create_const_nodes
from mo.graph.graph import Node
from mo.middle.passes import tensor_names, convert_data_type


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
        in_names = [node.in_node(id).id for id in range(len(node.in_nodes()))]
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
    while not node.has_valid('fw_tensor_debug_info') and len(node.in_nodes()):
        node = node.in_node()
    return node.soft_get('fw_tensor_debug_info')


def get_sorted_outputs(graph: nx.MultiDiGraph):
    outputs = []
    outputs_for_sort = {}
    for node in graph.nodes():
        if len(graph.out_edges(node)) == 0:
            outputs.append(Node(graph, node))
    if len(outputs) == 1:
        return outputs
    for node in outputs:
        outputs_for_sort[node.id] = get_fw_tensor_debug_info(node)
    return [Node(graph, key) for key, value in sorted(outputs_for_sort.items(), key=itemgetter(1))]


def prepare_emit_ir(graph: nx.MultiDiGraph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = []):
    create_const_nodes(graph)
    port_renumber(graph)

    op_order, data_order = determined_sort(get_sorted_outputs(graph))
    op_mapping = {v: u for u, v in enumerate(op_order)}
    data_mapping = {v: u for u, v in enumerate(data_order, start=len(graph.nodes()))}
    graph = nx.relabel_nodes(G=graph, mapping=op_mapping)
    graph = nx.relabel_nodes(G=graph, mapping=data_mapping)

    convert_data_type.convert(graph, data_type)
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
                   mean_size=mean_size)
    tensor_names.output_tensor_names_map(graph, os.path.join(output_dir, '{}.mapping'.format(output_model_name)))

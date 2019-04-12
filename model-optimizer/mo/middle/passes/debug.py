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

from mo.graph.graph import *


def print_attributes_in_col(attrs: dict, indent: int, exclude_attrs: list):
    for key in sorted(attrs.keys()):
        if key not in exclude_attrs:
            print(' ' * indent + key + ': ' + str(attrs[key]))


def debug_ir_emitter(graph, exclude_attrs: list = []):
    print("--- DEBUG IR BEGIN ---")
    # print nodes in topological order; print all attributes except 'pb'
    nodes = nx.topological_sort(graph)
    np.set_printoptions(threshold=10)
    for node in nodes:
        attrs = graph.node[node]
        print('Node:', node)
        if attrs['kind'] == 'op':
            for idx, value in Node(graph, node).in_nodes().items():
                print('input', idx, ':', value.node,
                      ': ' + str(graph.in_edges(value.node)[0][0]) if len(graph.in_edges(value.node)) else '')
        if 'op' in attrs:
            print('Op:', attrs['op'])
        print_attributes_in_col(attrs, 4, exclude_attrs)
        if attrs['kind'] == 'op':
            for idx, value in Node(graph, node).out_nodes().items():
                print('output', idx, ':', value.node,
                      ': ' + str(graph.out_edges(value.node)[0][1]) if len(graph.out_edges(value.node)) else '')
        print('')
    print("---  DEBUG IR END  ---")


def get_output_node_names(graph: Graph):
    result = []
    for node in graph.nodes():
        node = Node(graph, node)
        if len(node.out_nodes()) == 0:
            result.append(node.in_node().name)
    return result

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


from defusedxml.minidom import parseString
from xml.etree.ElementTree import Element, SubElement, tostring

from mo.graph.graph import Node, Graph


def propagate_op_name_to_tensor(graph: Graph):
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind == 'op' and node.has_valid('name'):
            for out_node, edge in node.out_nodes_edges().values():
                assert out_node.kind == 'data'
                out_node['ie_tensor_name'] = node.name
                out_node['ie_tensor_port'] = edge['out']
                out_node['ie_tensor_id'] = node.node


def output_tensor_names_map(graph: Graph, xml_file_name: str):
    mapping = Element('mapping')
    for node in graph:
        node = Node(graph, node)
        if node.has_valid('fw_tensor_debug_info') and node.has_valid('ie_tensor_name'):
            for fw_tensor_debug_info in node.fw_tensor_debug_info:
                # Check that debug info has valid fw attrs
                if not all(attr is not None for attr in fw_tensor_debug_info):
                    continue
                map = SubElement(mapping, 'map')
                fw = SubElement(map, 'framework')
                ie = SubElement(map, 'IR')

                fw.set('name', fw_tensor_debug_info[0])
                fw.set('out_port_id', str(fw_tensor_debug_info[1]))

                if node.has_valid('ie_tensor_name'):
                    ie.set('name', node.ie_tensor_name)
                if node.has_valid('ie_tensor_port'):
                    ie.set('out_port_id', str(node.ie_tensor_port))
                if node.has_valid('ie_tensor_id'):
                    ie.set('id', str(node.ie_tensor_id))
    with open(xml_file_name, 'w') as file:
        file.write(parseString(tostring(mapping)).toprettyxml())

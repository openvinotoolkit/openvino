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

import hashlib
import os
import xml.etree.ElementTree as ET
from collections import namedtuple, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import logging as log
import sys

from mo.graph.graph import Node, Graph
from mo.utils.unittest.graph import compare_graphs

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)

class IREngine(object):
    def __init__(self, path_to_xml: str, path_to_bin=None, precision="FP32", xml_tree=None):
        if not xml_tree and not os.path.exists(path_to_xml):
            raise AttributeError("File {} do not exists!".format(path_to_xml))

        if path_to_bin and not os.path.exists(path_to_bin):
            raise AttributeError("File {} do not exists!".format(path_to_bin))

        self.path_to_xml = str(path_to_xml)
        self.path_to_bin = str(path_to_bin) if path_to_bin else None
        self.xml_tree = xml_tree
        self.input_node = None

        if precision.upper() not in ['FP32', 'FP16']:
            raise AttributeError("Precision {} is not supported!".format(precision))
        self.__load_ir()

    def __load_xml(self):
        xml_tree = self.xml_tree or ET.parse(self.path_to_xml)
        xml_root = xml_tree.getroot()
        xml_layers = {}
        xml_edges = []
        statistics = {}

        Edge = namedtuple('edge', ['from_layer', 'from_port', 'to_layer', 'to_port'])

        # Create graph with operations only
        self.graph = Graph()
        self.graph.graph['hashes'] = {}

        # Parse XML
        for child in xml_root:
            if child.tag == 'layers':
                for layer in child:
                    layer_id, layer_attrs = self.__load_layer(layer)
                    xml_layers.update({layer_id: layer_attrs})
            elif child.tag == 'edges':
                for edge in child:
                    xml_edges.append(Edge(edge.attrib['from-layer'], int(edge.attrib['from-port']),
                                          edge.attrib['to-layer'], int(edge.attrib['to-port'])))
            elif child.tag == 'statistics':
                layers = child.findall('layer')
                for layer in layers:
                    statistics[layer.find('name').text] = {'min': layer.find('min').text, 'max': layer.find('max').text}

        self.graph.graph['statistics'] = statistics

        for layer in xml_layers.keys():
            self.graph.add_node(layer, **xml_layers[layer])

        for edge in xml_edges:
            self.graph.add_edges_from(
                [(edge.from_layer, edge.to_layer, {'from_port': edge.from_port, 'to_port': edge.to_port})])

        # Insert data nodes between op nodes and insert data nodes with weights
        nodes = list(self.graph.nodes())
        for node in nodes:
            out_edges = Node(self.graph, node).get_outputs()
            data_nodes = {}
            for port in self.graph.node[node]['ports']:
                data = self.graph.unique_id(prefix='data_')
                self.graph.add_node(data, **{'kind': 'data', 'shape': self.graph.node[node]['ports'][port],
                                             'value': None})
                self.graph.add_edges_from([(node, data, {'out': port})])
                data_nodes.update({port: data})

            for out_node, edge_attrs in out_edges:
                self.graph.remove_edge(node, out_node)
                if edge_attrs['from_port'] in data_nodes:
                    data = data_nodes[edge_attrs['from_port']]
                else:
                    raise RuntimeError("SMTH wrong with IR! There is an edge from not existing port")
                self.graph.add_edges_from([(data, out_node, {'in': edge_attrs['to_port']})])

    def __load_bin(self):
        bin_buff = np.fromfile(file=self.path_to_bin, dtype=np.uint8)
        graph = self.graph
        nodes = [node for node in graph.nodes()]
        hashes = defaultdict(dict)
        for node in nodes:
            for w in ['weights', 'biases', 'custom']:
                if w in graph.node[node]:
                    data = graph.unique_id(prefix='data_')
                    offset, size, in_port, precision = graph.node[node][w]
                    if Node(graph, node).soft_get('type') == 'BinaryConvolution':
                        precision = np.uint8
                    value = np.frombuffer(buffer=bin_buff, dtype=precision, count=size, offset=offset)
                    hashes[graph.node[node]['name']][w] = hashlib.sha512(value.tobytes()).hexdigest()
                    graph.add_node(data, **{'kind': 'data', 'value': value, 'shape': value.shape})
                    graph.add_edges_from([(data, node, {'in': in_port})])
        self.graph.graph['hashes'].update(hashes)

    def __load_bin_hashes(self):
        graph = self.graph
        bin_hash_map = {name: blob_map.item(0) for name, blob_map in dict(np.load(self.path_to_bin,
                                                                                  allow_pickle=True)).items()}

        for node in graph.nodes():
            for w in ['weights', 'biases', 'custom']:
                if w in graph.node[node]:
                    assert Node(graph, node).has_valid('name')
                    node_name = Node(graph, node).name
                    assert node_name in bin_hash_map and w in bin_hash_map[node_name]
                    graph.node[node]['hashes'] = bin_hash_map[node_name][w]


    def __load_ir(self):
        self.__load_xml()
        if not self.path_to_bin:
            return

        if self.path_to_bin.endswith('.bin.hashes.npz'):
            self.__load_bin_hashes()
        else:
            self.__load_bin()

    def __load_layer(self, layer):
        """
            Layer example

            <layer id="1" name="862" precision="FP32" type="Convolution">
                <data dilation-x="1" dilation-y="1" group="1" kernel-x="1" kernel-y="5" output="32" pad-b="0" pad-r="2" pad-x="2" pad-y="0" stride-x="1" stride-y="1"/>
                <input>
                    <port id="0">
                        <dim>1</dim>
                        <dim>3</dim>
                        <dim>32</dim>
                        <dim>32</dim>
                    </port>
                </input>
                <output>
                    <port id="3">
                        <dim>1</dim>
                        <dim>32</dim>
                        <dim>32</dim>
                        <dim>32</dim>
                    </port>
                </output>
                <blobs>
                    <weights offset="0" size="1920"/>
                    <biases offset="1920" size="128"/>
                </blobs>
            </layer>

        """

        layer_id = layer.attrib['id']

        layer_attrs = layer.attrib
        layer_attrs.update({'ports': {}, 'kind': 'op'})

        inputs_counter = 0

        for attr in layer:
            if attr.tag == 'data':
                layer_attrs.update(IREngine.__normalize_attrs(attr.attrib))
            elif attr.tag == 'input':
                inputs_counter = len(attr)
            elif attr.tag == 'output':
                output = attr
                for port in output:
                    port_id = int(port.attrib['id'])
                    output_shape = []
                    for dim in port:
                        output_shape.append(int(dim.text))

                    layer_attrs['ports'].update({port_id: output_shape})
            elif attr.tag == 'blobs':
                in_port = inputs_counter
                precision = layer.attrib['precision']
                precision_map = {
                    'FP32': (4, np.float32),
                    'FP16': (2, np.float16),
                    'I64': (8, np.int64),
                    'I32': (4, np.int32),
                }
                type_size, dtype = precision_map[precision]
                for blob_attr in attr:
                    layer_attrs.update({blob_attr.tag: (int(blob_attr.attrib['offset']),
                                                        int(blob_attr.attrib['size']) // type_size,
                                                        in_port,
                                                        dtype)})
                    in_port += 1
            elif attr.tag == 'body':
                xml_body_child = list(layer.iterfind('body'))
                assert len(xml_body_child) == 1

                body_ir = IREngine(path_to_xml=None,
                                   path_to_bin=self.path_to_bin,
                                   xml_tree=ET.ElementTree(xml_body_child[0]))
                self.graph.graph['hashes'].update(body_ir.graph.graph['hashes'])

                # Find port_map section and take an input with axis specified - this will be out input_layer for body
                xml_port_map = list(layer.iterfind('port_map'))
                if not len(xml_port_map) == 1:
                    log.warning("TensorIterator body won\'t be compared due to missing port_map section!")
                    continue
                xml_port_map = xml_port_map[0]

                input_layers = []
                for input in xml_port_map:
                    if input.tag == 'input' and 'axis' in input.attrib:
                        if 'internal_layer_id' not in input.attrib:
                            log.warning("internal_layer_id attrib not found in input section")
                        else:
                            input_layers.append(Node(body_ir.graph, input.attrib['internal_layer_id']))

                if len(input_layers) != 1:
                    log.warning("TensorIterator body won\'t be compared due to the number of inputs in body != 1 "
                                "({})".format(len(input_layers)))
                else:
                    body_ir.input_node = input_layers[0]
                    layer_attrs.update({'body': body_ir})

        return layer_id, layer_attrs

    @staticmethod
    def __normalize_attrs(attrs: dict):
        """
        Normalize attributes for type 'data'.
        Replace " from values (not used right now) and make list of value with int, float or other types values.
        Example: {'order': '1,0,2'} -> {'order': [1, 0, 2]}
                 {'order': '1'}     -> {'order': 1}
        """
        normalized_attrs = {}
        for attr, value in attrs.items():
            value = value.replace('\"', '')
            value = value.split(',')
            n_value = []
            for val in value:
                if val.isdigit():
                    n_value.append(int(val))
                elif IREngine.__isfloat(val):
                    n_value.append(float(val))
                else:
                    n_value.append(val)

            if len(n_value) == 1:
                normalized_attrs.update({attr: n_value[0]})
            else:
                normalized_attrs.update({attr: n_value})

        return normalized_attrs

    @staticmethod
    def __isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def __find_input(graph):
        inputs = []
        for node in sorted(graph.nodes()):
            node = Node(graph, node)
            if node.has_valid('type') and node.type == 'Input':
                inputs.append(node)

        if len(inputs) < 1:
            raise RuntimeError("Graph {} has less than one input node")

        return inputs

    def compare(self, ref_net):
        if not isinstance(ref_net, IREngine):
            ir_input = self.__find_input(self.graph)[0]
            ref_input = self.__find_input(ref_net)[0]
            ref_graph = ref_net
        else:
            ir_input = self.input_node or self.__find_input(self.graph)[0]
            ref_input = ref_net.input_node or ref_net.__find_input(ref_net.graph)[0]
            ref_graph = ref_net.graph
        # TODO check that ir_input[0].id and ref_input[0].id are the same
        result, stderr = compare_graphs(graph=self.graph, graph_ref=ref_graph, last_node=ir_input.id,
                                        last_node_ref=ref_input.id, check_op_attrs=True)
        return result, stderr

    def generate_bin_hashes_file(self, path_for_file=None):
        # This function creates file with extension '.bin.hashes.npz' where hashes of bin exists.
        # For creating this file in custom filder use attribute path_for_file.
        # Where directory for file should be existed
        graph = self.graph
        if path_for_file is None:
            path_for_file = str(Path(self.path_to_xml).with_suffix('.bin.hashes.npz'))
        assert 'hashes' in graph.graph, "Loaded IR graph doesn't contain `hashes`: {}".format(self.path_to_xml)
        np.savez_compressed(path_for_file, **graph.graph['hashes'])
        return path_for_file

    def get_inputs(self):
        # Function return input nodes in dictionary: {input_node_name: input_node_shape, ...}
        input_nodes = self.__find_input(self.graph)
        return {input_node.name: input_node.out_node().shape for input_node in input_nodes}

    def __eq__(self, other):
        # To call this function create two IREngine objects (IR1, IR2) and compare them IR1 == IR2
        if not isinstance(other, IREngine):
            raise AttributeError("IREngine can be compared only with IREngine object type")
        return self.compare(other)[0]
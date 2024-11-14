# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging as log
import os
import sys
from argparse import Namespace
from collections import namedtuple, defaultdict, OrderedDict
from pathlib import Path

import defusedxml.ElementTree as ET
import numpy as np
from defusedxml import defuse_stdlib

from openvino.tools.mo.front.common.partial_infer.utils import dynamic_dimension_value, shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from openvino.tools.mo.utils.runtime_info import RTInfo, OldAPIMapOrder, OldAPIMapElementType

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.DEBUG, stream=sys.stdout)

# defuse_stdlib provide patched version of xml.etree.ElementTree which allows to use objects from xml.etree.ElementTree
# in a safe manner without including unsafe xml.etree.ElementTree
ElementTree = defuse_stdlib()[ET].ElementTree


def read_rt_info_attr(elem):
    if len(elem) == 0:
        if 'value' not in elem.attrib:
            return None
        value = elem.attrib['value']
        return value
    val_dict = {}
    for child in elem:
        child_val = read_rt_info_attr(child)
        val_dict[child.attrib.get('name', child.tag)] = child_val
    return val_dict


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
        self.ir_version = None
        self.meta_data = dict()

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

        self.graph.graph['ir_version'] = int(xml_root.attrib['version']) if xml_root.attrib.get('version') is not None else None

        # NOTE: THis is MO internal attribute, it cannot be used for
        # defining graph input layout. We set it to NCHW as in MO back stage
        # during conversion for correct shape inference of layout specific
        # operations (ExtractImagePatches, SpaceToDepth, etc.)
        self.graph.graph['layout'] = 'NCHW'

        self.graph.name = xml_root.attrib['name'] if xml_root.attrib.get('name') is not None else None

        self.graph.inputs_order = []
        self.graph.outputs_order = []

        # Parse XML
        for child in xml_root:
            if child.tag == 'layers':
                for layer in child:
                    layer_id, layer_attrs = self.__load_layer(layer)
                    xml_layers.update({layer_id: layer_attrs})
                    if layer_attrs['type'] == 'Parameter':
                        self.graph.inputs_order.append(layer_attrs['name'])
                    if layer_attrs['type'] == 'Result':
                        self.graph.outputs_order.append(layer_attrs['name'])
            elif child.tag == 'edges':
                for edge in child:
                    xml_edges.append(Edge(edge.attrib['from-layer'], int(edge.attrib['from-port']),
                                          edge.attrib['to-layer'], int(edge.attrib['to-port'])))
            elif child.tag == 'statistics':
                layers = child.findall('layer')
                for layer in layers:
                    statistics[layer.find('name').text] = {'min': layer.find('min').text, 'max': layer.find('max').text}
            elif child.tag == 'rt_info':
                for elem in child:
                    self.meta_data[elem.attrib.get('name', elem.tag)] = read_rt_info_attr(elem)

            # TODO: Remove this part when POT updates to using of rt_info
            elif child.tag == 'quantization_parameters':
                # Section with Post Optimization Toolkit parameters
                self.meta_data['quantization_parameters'] = dict()
                for elem in child:
                    if elem.tag == 'config':
                        self.meta_data['quantization_parameters']['config'] = elem.text
                    elif elem.tag in ['version', 'cli_params']:
                        self.meta_data['quantization_parameters'][elem.tag] = elem.attrib['value']

        self.graph.graph['cmd_params'] = Namespace(**self.meta_data)  # TODO check what we need all this attrs

        if len(statistics):
            self.graph.graph['statistics'] = statistics

        for layer in xml_layers.keys():
            self.graph.add_node(layer, **xml_layers[layer])

        xml_edges.sort(key=lambda x: x.to_layer)

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
                self.graph.add_node(data, **{'kind': 'data', 'shape': self.graph.node[node]['ports'][port][0],
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
        layer_attrs.update({'ports': {}, 'restored_input_ports': {}, 'kind': 'op'})

        inputs_counter = 0

        for attr in layer:
            if attr.tag == 'data':
                new_attrs = self.__normalize_attrs(attr.attrib)
                new_attrs['ir_data_attrs'] = attr.attrib
                if layer.attrib['type'] == 'Const':
                    assert 'offset' in new_attrs and 'size' in new_attrs, \
                        'Incorrect attributes for Const layer, {} instead of {}!'.format(new_attrs.keys(), ['offset', 'size'])
                    precision = ""
                    for item in layer:
                        if item.tag == "output":
                            precision = item[0].attrib["precision"]
                            break
                    new_attrs.update(self.__prepare_bin_attrs(layer, 0, 'custom', new_attrs['offset'], new_attrs['size'], precision))
                layer_attrs.update(new_attrs)
            elif attr.tag == 'input':
                inputs_counter = len(attr)

                input = attr
                for port in input:
                    port_id = int(port.attrib['id'])
                    input_shape = []
                    port_rt_info = {}
                    for dim in port:
                        if dim.tag == "dim":
                            input_shape.append(int(dim.text))
                        if dim.tag == 'rt_info':
                            for attr in dim:
                                port_rt_info.update(self.__read_rt_info_common(attr))

                    input_shape = shape_array([d if d != -1 else dynamic_dimension_value for d in input_shape])

                    in_tensor_names = None
                    if 'names' in port.attrib:
                        in_tensor_names = port.attrib['names']

                    # special attribute to pass information about operation input ports
                    layer_attrs['restored_input_ports'].update({port_id: (input_shape, in_tensor_names, port_rt_info)})
            elif attr.tag == 'output':
                output = attr
                for port in output:
                    port_id = int(port.attrib['id'])
                    output_shape = []
                    port_rt_info = {}
                    for dim in port:
                        if dim.tag == "dim":
                            output_shape.append(int(dim.text))
                        if dim.tag == 'rt_info':
                            for attr in dim:
                                port_rt_info.update(self.__read_rt_info_common(attr))

                    output_shape = shape_array([d if d != -1 else dynamic_dimension_value for d in output_shape])

                    out_tensor_names = None
                    if 'names' in port.attrib:
                        out_tensor_names = port.attrib['names']
                    # special attribute to pass information about operation input ports
                    # NOTE: renaming or structure changing of this attribute may have big impact on tests
                    layer_attrs['ports'].update({port_id: (output_shape, out_tensor_names, port_rt_info)})
            elif attr.tag == 'blobs':
                in_port = inputs_counter
                for blob_attr in attr:
                    layer_attrs.update(self.__prepare_bin_attrs(layer, in_port, blob_attr.tag,
                                                                blob_attr.attrib['offset'], blob_attr.attrib['size'],
                                                                blob_attr.attrib.get('precision', None)))
                    in_port += 1
            elif attr.tag == 'body':
                xml_body_child = list(layer.iterfind('body'))
                assert len(xml_body_child) == 1

                body_ir, input_port_map, output_port_map, input_layers = \
                    self.__read_subgraph(layer, layer_attrs, xml_body_child, 'port_map')

                body_ir.input_node = input_layers[0]
                layer_attrs.update({'body': body_ir})
                layer_attrs.update({'input_port_map': input_port_map})
                layer_attrs.update({'output_port_map': output_port_map})

                xml_back_edges_map = list(layer.iterfind('back_edges'))
                if not len(xml_back_edges_map) == 1:
                    log.warning("TensorIterator body won\'t be compared due to missing back_edges section!")
                    continue
                xml_back_edges_map = xml_back_edges_map[0]

                back_edges = []

                for edge in xml_back_edges_map:
                    back_edges.append(self.__normalize_attrs(edge.attrib))

                layer_attrs.update({'back_edges': back_edges})

            elif attr.tag == 'then_body' or attr.tag == 'else_body':
                assert layer.attrib['type'] == 'If', "Incorrect IR! The operation {0}" \
                                                     " has sub-graphs for If operation"
                layer_attrs = self.__read_if(layer, layer_attrs)
                continue

            elif attr.tag == 'rt_info':
                layer_attrs = self.__read_rt_info(layer, layer_attrs)
                continue

        return layer_id, layer_attrs

    @staticmethod
    def __prepare_bin_attrs(xml_layer, in_port, tag, offset, size, precision):
        layer_attrs = dict()
        if precision is None:
            precision = xml_layer.attrib['precision']
        precision_map = {
            'FP64': (8, np.float64),
            'FP32': (4, np.float32),
            'FP16': (2, np.float16),
            'I64': (8, np.int64),
            'I32': (4, np.int32),
            'I8': (1, np.int8),
            'U8': (1, np.uint8),
            'U1': (1, np.uint8),
            'U4': (1, np.uint8),
            'I4': (1, np.uint8),
            'BOOL': (1, bool),
            'BIN': (1, np.uint8),
            'U64': (8, np.uint64)
        }
        type_size, dtype = precision_map[precision]
        layer_attrs[tag] = (int(offset), int(size) // type_size, in_port, dtype)
        return layer_attrs

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
            value = value.replace('\"', '').replace(' ', '')
            value = value.split(',')
            n_value = []
            for val in value:
                if IREngine.__isint(val):
                    n_value.append(int(val))
                elif IREngine.__isfloat(val):
                    n_value.append(float(val))
                elif val in ['True', 'False', 'true', 'false']:
                    n_value.append(val in ['True', 'true'])
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
    def __isint(value):
        is_signed = value.startswith('+') or value.startswith('-')
        other_chars_are_digits = value[1:].isdigit()
        all_chars_are_digits = value.isdigit()
        return all_chars_are_digits or (is_signed and other_chars_are_digits)

    @staticmethod
    def __find_input(graph):
        inputs = []
        for node in sorted(graph.nodes()):
            node = Node(graph, node)
            if node.has_valid('type') and node.type in ('Input', 'Parameter'):
                inputs.append(node)

        if len(inputs) < 1:
            raise RuntimeError("Graph {} has less than one input node".format(graph.name))

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

    def __read_subgraph(self, layer, layer_attrs, body_child, port_map_name):
        body_ir = IREngine(path_to_xml=None,
                           path_to_bin=self.path_to_bin,
                           xml_tree=ElementTree(body_child[0]))

        self.graph.graph['hashes'].update(body_ir.graph.graph['hashes'])

        xml_port_map = list(layer.iterfind(port_map_name))
        assert not len(xml_port_map) != 1, "If then_body won\'t be compared due to missing {1} section in node {0}! " \
            .format(layer_attrs['name'], port_map_name)
        xml_port_map = xml_port_map[0]

        input_layers = []
        input_port_map = []
        output_port_map = []

        for port in xml_port_map:
            if port.tag == 'input':
                if 'internal_layer_id' not in port.attrib:
                    log.warning("internal_layer_id attrib not found in input section")
                else:
                    input_layers.append(Node(body_ir.graph, port.attrib['internal_layer_id']))
                    input_port_map.append(self.__normalize_attrs(port.attrib))
            elif port.tag == 'output':
                if 'internal_layer_id' not in port.attrib:
                    log.warning("internal_layer_id attrib not found in output section")
                else:
                    output_port_map.append(self.__normalize_attrs(port.attrib))

        return body_ir, input_port_map, output_port_map, input_layers

    def __read_if(self, layer, layer_attrs):

        xml_then_body_child = list(layer.iterfind('then_body'))
        xml_else_body_child = list(layer.iterfind('else_body'))
        assert len(xml_then_body_child) == 1 and len(xml_else_body_child) == 1, "If operation has only one subgraph"

        then_body_ir, then_input_port_map, then_output_port_map, _ = \
            self.__read_subgraph(layer, layer_attrs, xml_then_body_child, 'then_port_map')
        layer_attrs.update({'then_graph': then_body_ir})
        layer_attrs.update({'then_input_port_map': then_input_port_map})
        layer_attrs.update({'then_output_port_map': then_output_port_map})

        else_body_ir, else_input_port_map, else_output_port_map, _ = \
            self.__read_subgraph(layer, layer_attrs, xml_else_body_child, 'else_port_map')
        layer_attrs.update({'else_graph': else_body_ir})
        layer_attrs.update({'else_input_port_map': else_input_port_map})
        layer_attrs.update({'else_output_port_map': else_output_port_map})

        return layer_attrs

    def __read_rt_info(self, layer, layer_attrs):
        rt_info = RTInfo()
        xml_rt_info = list(layer.iterfind('rt_info'))[0]

        for attr in xml_rt_info:
            attr_name = attr.attrib['name']
            if attr_name == 'old_api_map_order':
                rt_info.info.update(self.__read_old_api_map_order(attr, layer.attrib['type']))
            elif attr_name == 'old_api_map_element_type':
                rt_info.info.update(self.__read_old_api_map_element_type(attr, layer.attrib['type']))
            else:
                rt_info.info.update((self.__read_rt_info_common(attr)))

        layer_attrs.update({'rt_info': rt_info})
        return layer_attrs

    @staticmethod
    def __read_old_api_map_order(attr, layer_type):
        version = int(attr.attrib['version'])
        order = list(map(int, attr.attrib['value'].split(',')))
        old_api_map = OldAPIMapOrder(version=version)
        if layer_type == 'Parameter':
            old_api_map.old_api_transpose_parameter(order)
        elif layer_type == 'Result':
            old_api_map.old_api_transpose_result(order)
        else:
            raise AttributeError("Cannot read old_api_map for layer of type: {}".format(layer_type))

        return {('old_api_map_order', version): old_api_map}

    @staticmethod
    def __read_old_api_map_element_type(attr, layer_type):
        version = int(attr.attrib['version'])
        element_type = destination_type_to_np_data_type(attr.attrib['value'])
        old_api_map = OldAPIMapElementType(version=version)
        old_api_map.set_legacy_type(element_type)
        return {('old_api_map_element_type', version): old_api_map}

    @staticmethod
    def __read_rt_info_common(attr):
        attr_name = attr.attrib['name']
        version = int(attr.attrib['version'])
        rt_info = OrderedDict()
        for key in attr.attrib:
            if key not in ('name', 'version'):
                rt_info[key] = attr.attrib[key]
        return {(attr_name, version): rt_info}

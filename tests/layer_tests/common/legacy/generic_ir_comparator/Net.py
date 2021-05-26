import os
import re
import logging as log
from xml.etree.ElementTree import Element, SubElement, ElementTree

import networkx as nx
import numpy as np

from common.legacy.generic_ir_comparator.IR import IR
from common.legacy.generic_ir_comparator.layers import *


try:
    import constants
except (SystemError, ImportError):
    constants = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

class Network:
    def __init__(self, name='network', precision='FP32'):
        self.layer_id = 0
        self.port_id = 0
        self.offset = 0
        self.name = name
        self.precision = precision
        self.topology = nx.DiGraph()

    def add_layer(self, layer_type, layer_name=None, inputs=None, get_out_shape_def=None,
                  framework_representation_def=None, ir_representation_def=None, **layer_param):
        input_layers = {}
        if inputs:
            for input_layer in inputs:
                input_layers[input_layer.get_name()] = input_layer.get_out_shapes()

        layer = Layer(layer_type=layer_type,
                      layer_name=layer_name,
                      layer_id=self.layer_id,
                      port_id=self.port_id,
                      precision=self.precision,
                      inputs=input_layers,
                      attrs=layer_param,
                      get_out_shape_def=get_out_shape_def,
                      framework_representation_def=framework_representation_def,
                      ir_representation_def=ir_representation_def)

        self.layer_id += 1
        self.port_id = layer.get_out_port_ids()[-1] + 1
        self.topology.add_node(layer, checked=False)
        if inputs:
            for input_layer in inputs:
                self.topology.add_edge(input_layer, layer)
        return layer

    def load_xml(self, xml_tree: ElementTree):
        self.layer_id = 0
        self.port_id = 0
        self.offset = 0
        root = xml_tree.getroot()
        self.name = root.get('name')
        edges = list(root.find('edges'))
        layers = list(root.find('layers'))
        self.topology = nx.DiGraph()
        for layer in layers:
            self.topology.add_node(Layer(xml_element=layer), checked=False)
        for edge in edges:
            from_layer = self.find_layer_by_id(int(edge.get('from-layer')))
            to_layer = self.find_layer_by_id(int(edge.get('to-layer')))
            to_layer.set_input(from_layer)
            if from_layer and to_layer:
                self.topology.add_edge(from_layer, to_layer)

    def load_bin(self, filename):
        method_to_call = 'float32' if self.precision == 'FP32' else 'float16'
        bin_array = np.fromfile(file=filename, dtype=getattr(np, method_to_call))
        for layer in self.topology.nodes():
            layer.load_bin(bin_array)

    def compare(self, other, skip_layers_types_pattern='^$', break_on_first_diff=True, ignore_attributes=None):
        """
        :param ignore_attributes: dict of attributes to ignore{'Split':['num_split']}
        :param other: layer is used to compare with self
        :param skip_layers_types_pattern: Pattern to skip layer comparing
        :param break_on_first_diff: if False show all differences between nets
        :return: True if nets are equal
        """
        status = True

        log.debug('Skipping comparing pairs of layers mach pattern: {}'.format(skip_layers_types_pattern))
        log.info('Ref net contains {} edges and {} layers.'.format(len(self.topology.edges()),
                                                                   len(self.topology.nodes())))
        if len(self.topology.edges()) != len(other.topology.edges()):
            log.warning(
                'Different number of edges: {} vs {}'.format(len(self.topology.edges()), len(other.topology.edges())))

        if len(self.topology.nodes()) != len(other.topology.nodes()):
            log.warning(
                'Different number of layers: {} vs {}'.format(len(self.topology.nodes()), len(other.topology.nodes())))

        if not ignore_attributes:
            ignore_attributes = {}

        inputs_self = self.get_inputs()
        inputs_others = other.get_inputs()

        # sort inputs
        inputs_self = tuple(sorted(inputs_self, key=lambda i: i.name))
        inputs_others = tuple(sorted(inputs_others, key=lambda i: i.name))

        status = status and self.compare_nodes(inputs_self, inputs_others, other.topology, skip_layers_types_pattern,
                                               break_on_first_diff, ignore_attributes)
        return status

    def compare_nodes(self, nodes_self: tuple, nodes_other: tuple, other_graph: nx.DiGraph,
                      skip_layers_types_pattern: str, break_on_first_diff: bool, ignore_attributes: dict):
        status = True
        pattern = re.compile(skip_layers_types_pattern)
        for node_self in nodes_self:
            if pattern.match(node_self.type):
                nodes_tmp = tuple(
                    u for u in nodes_other if (
                            u.type == node_self.type or pattern.match(u.type)) and not Network.is_checked_node(
                        other_graph, u))
            else:
                nodes_tmp = tuple(
                    u for u in nodes_other if u.type == node_self.type and not Network.is_checked_node(other_graph, u))

            if nodes_tmp and not Network.is_checked_node(self.topology, node_self):
                for node_other in nodes_tmp:
                    if (pattern.match(node_other.type) and pattern.match(node_self.type)) or node_self.compare(
                            node_other, break_on_first_diff, ignore_attributes):
                        Network.set_checked_for_node(self.topology, node_self, True)
                        Network.set_checked_for_node(other_graph, node_other, True)

                        children_self = Network.find_all_not_checked_children(node_self, self.topology)
                        children_other = Network.find_all_not_checked_children(node_other, other_graph)
                        if len(children_self) == len(children_other):
                            status = status and self.compare_nodes(children_self, children_other, other_graph,
                                                                   skip_layers_types_pattern, break_on_first_diff,
                                                                   ignore_attributes)
                        break
                else:
                    log.error("{}: matcher for '{}' not found!".format(node_self.type, node_self.get_name()))
                    status = False
        return status

    @staticmethod
    def is_checked_node(graph: nx.DiGraph, node):
        for n in graph.nodes(data=True):
            if n[0] == node:
                return n[1]['checked']

    @staticmethod
    def set_checked_for_node(graph: nx.DiGraph, node, checked):
        for n in graph.nodes(data=True):
            if n[0] == node:
                n[1]['checked'] = checked
                break
        else:
            raise RuntimeError

    @staticmethod
    def find_all_not_checked_children(node, graph: nx.DiGraph):
        return set(e[1] for e in graph.edges(node) if not Network.is_checked_node(graph, e[1]))

    def find_layer_by_id(self, layer_id: int):
        for layer in self.topology.nodes():
            if layer.id == layer_id:
                return layer

    def get_name(self):
        return self.name

    def get_inputs(self):
        inputs = set()
        for n in self.topology.nodes(data=False):
            for e in self.topology.edges():
                if e[1] == n:  # n is child
                    break
            else:
                inputs.add(n)
        return inputs

    def get_input_name(self):
        return [u for u in self.get_inputs()][0].get_name()

    def get_input_shape(self):
        return [u for u in self.get_inputs()][0].get_out_shapes()[0]

    def get_outputs_name(self):
        outputs = []
        for u in self.topology.nodes(data=False):
            if len(self.topology.edges(u)) == 0:
                outputs.append(u.get_name())
        return outputs

    def get_children_proto(self, inputs, v=None):
        nodes = []
        t = set()

        if not v:
            v = set()

        for input_layer in inputs:
            nodes += [input_layer.framework_representation_def(input_layer)]
            v.add(input_layer)
        for e in self.topology.edges(inputs):
            t.add(e[1])
        inputs = set()
        tmp = set()
        for n in t:
            for e in self.topology.in_edges(n):
                if e[0] in v:
                    inputs.add(e[1])
                else:
                    tmp.add(e[1])
        inputs -= tmp
        if inputs:
            nodes += self.get_children_proto(inputs - tmp, v)
        return nodes

    def save_ir(self, path, name='net'):
        method_to_call = 'float32' if self.precision == 'FP32' else 'float16'

        xml, bin = self.get_ir()
        IR(xml).save(os.path.join(path, name + '.xml'))
        np.array(bin, getattr(np, method_to_call)).tofile(os.path.join(path, name + '.bin'))

    def get_ir(self, version=2):
        net = Element('net')
        net.set('name', self.name)
        net.set('version', str(version))
        net.set('batch', str(1))
        layers_xml = SubElement(net, 'layers')
        edges_xml = SubElement(net, 'edges')
        inputs = self.get_inputs()
        bin = self.get_children_xml(inputs, layers_xml, edges_xml)
        self.offset = 0  # offset is calculated every time anew during the construction xml
        return ElementTree(net), bin

    def get_children_xml(self, inputs, layers_xml, edges_xml, v=set(), bin=list()):
        t = set()
        for input_layer in inputs:
            layers_xml.insert(input_layer.id, input_layer.to_xml(self.precision, self.offset))
            input_layer.get_bin(bin)
            v.add(input_layer)
            self.offset = input_layer.get_size_weights()
            for e in self.topology.in_edges(input_layer):
                el = SubElement(edges_xml, 'edge')
                el.set('from-layer', str(e[0].id))
                el.set('from-port', str(e[0].get_out_port_ids()[0]))
                el.set('to-layer', str(input_layer.id))
                el.set('to-port', str(input_layer.get_in_port_id(e[0].get_out_name())[0]))
        for e in self.topology.edges(inputs):
            t.add(e[1])
        inputs = set()
        tmp = set()
        for n in t:
            for e in self.topology.in_edges(n):
                if e[0] in v:
                    inputs.add(e[1])
                else:
                    tmp.add(e[1])
        inputs -= tmp
        if inputs:
            self.get_children_xml(inputs - tmp, layers_xml, edges_xml, v)
        return bin

    def generate_tf_pb(self, path=None, name=None, as_text=False, output=None):
        if not path:
            path = constants.tf_models_path

        if not os.path.exists(path):
            os.mkdir(path)

        if not name:
            name = '{}.pb'.format(self.name)

        inputs = self.get_inputs()
        with tf.compat.v1.Session() as sess:
            for input_layer in inputs:
                self.add_node_to_tf_graph(input_layer)
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess, tf.compat.v1.get_default_graph().as_graph_def(),
                self.get_outputs_name() if not output else output)
            tf.io.write_graph(output_graph_def, path, name, as_text=as_text)
            tf.io.write_graph(output_graph_def, path, name + '.txt', as_text=True)
        tf.compat.v1.reset_default_graph()

    def add_node_to_tf_graph(self, layer, input_tensor_name=None, tf_graph_inputs=None):
        if not tf_graph_inputs:
            tf_graph_inputs = {}

        input_tensor = layer.framework_representation_def(layer, input_tensor_name, tf_graph_inputs)

        if input_tensor is not None:
            if isinstance(input_tensor, list) or isinstance(input_tensor, tuple):  # list or tuple of tensors
                for tensor in input_tensor:
                    tf_graph_inputs[tensor.name] = tensor
            else:  # Tensor
                tf_graph_inputs[input_tensor.name] = input_tensor

        for edge in self.topology.edges(layer):
            edge_inputs_set = set()
            for key, value in edge[1].inputs.items():
                for i in range(0, len(value)):
                    edge_inputs_set.add("{}:{}".format(key, i))

            if set(tf_graph_inputs.keys()) >= edge_inputs_set:  # if we have all input tensors
                if len(edge_inputs_set) == 1:
                    self.add_node_to_tf_graph(edge[1], edge_inputs_set.pop(), tf_graph_inputs)
                else:
                    self.add_node_to_tf_graph(edge[1], edge_inputs_set, tf_graph_inputs)

    def create_topology(self):
        # sort topology edges by levels
        sorted_edges = sorted(self.topology.edges(), key=lambda item: (item[1].id, -item[0].id))

        # add input layers
        layers = []
        for node in self.topology.nodes(data=False):
            if node.inputs:
                continue
            else:
                if node not in layers:
                    layers.append(node)
        # sort input layers in order to start from the left input to right
        layers = sorted(layers, key=lambda item: item.id)

        # delete all layers with outputs. All output layers will remain
        layers_to_delete = []
        params = dict()
        for layer in layers:

            layer_inputs = []
            layer_outputs = []

            for edge in sorted_edges:
                if edge[1] is layer and edge[0] not in layer_inputs:
                    layer_inputs.append(edge[0])
                if edge[0] is layer and edge[1] not in layer_outputs:
                    layer_outputs.append(edge[1])
                    # add another layers from layer output
                    if edge[1] not in layers:
                        layers.append(edge[1])

            # sort layer_inputs in order to start from lower levels to the upper
            layer_inputs = sorted(layer_inputs, key=lambda item: -item.id)

            layer.framework_representation_def = layer.framework_representation_def(layer, layer_inputs)

            if hasattr(layer, "params"):
                params.update(layer.params)
            if layer_outputs:
                layers_to_delete.append(layer)

        for i in layers_to_delete:
            layers.remove(i)

        return mx.symbol.Group([l.framework_representation_def for l in layers]), params

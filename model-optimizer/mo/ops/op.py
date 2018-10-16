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

import networkx as nx
import numpy as np

from mo.front.extractor import add_attrs_props
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Node, unique_id
from mo.utils import class_registration
from mo.utils.error import Error


class Op(object):
    registered_ops = {}
    registered_cls = []
    # Add the derived class to excluded_classes if one should not be registered in registered_ops
    excluded_classes = []

    def __init__(self, graph: nx.MultiDiGraph, attrs1: dict = None, attrs2: dict = None):
        self.graph = graph
        self.attrs = {
            'precision': "FP32",
            'kind': 'op'
        }
        self.default_backend_attrs = []
        if attrs1 is not None:
            self.attrs.update(attrs1)
        if attrs2 is not None:
            self.attrs.update(attrs2)

    def add_node(self, attrs: dict = None):
        new_attrs = {}
        new_attrs.update(self.attrs)
        if attrs is not None:
            new_attrs.update(attrs)
        id_prefix = new_attrs['name'] if 'name' in new_attrs else ''
        id = unique_id(self.graph, id_prefix)
        new_attrs['name'] = id
        new_attrs = add_attrs_props(new_attrs)
        update_ie_fields(new_attrs)
        self.substitute_ie_attrs(new_attrs)
        self.graph.add_node(id, **new_attrs)
        return Node(self.graph, id)

    def substitute_ie_attrs(self, new_attrs: dict):
        """
        Replace standard list of attribute in layer/data by attributes
        delivered by backend_attrs
        """

        new_attrs.update({
            'IE': [(
                'layer',
                [('id', lambda node: node.node), 'name', 'precision', 'type'],
                [
                    ('data', self.backend_attrs() + self.default_backend_attrs, []),
                    '@ports',
                    '@consts'])]
        })

    @staticmethod
    def extract_port(node_port):
        if isinstance(node_port, tuple):
            node = node_port[0]
            port = node_port[1]
        else:
            node = node_port
            port = 0
        # 'data' nodes do not have 'out' edge attibute but always has one output
        out_ids = [attr['out'] for _, __, attr in node.graph.out_edges(node.id, data=True) if 'out' in attr]
        if len(set(out_ids)) > 1 and not isinstance(node_port, tuple):
            raise Error('Node {} has more than one outputs. Provide output port explicitly. '.format(node.name))
        return node, port

    def cut_edge_and_create_node(self, node: Node, out_port: int, attrs: dict = None):
        """
        Removes an edge, that is connected to nodes out_port. Creates new_node with attrs attributes and
        connects it to node by edge that stores the same information as cutted edge.
        :param node: Input node, to cut the edge from
        :param out_port: output port of edge to cut
        :param attrs: attributes of new node
        :return: Node instance of created new_node
        """
        edges = [(u, v, keys, params) for u, v, keys, params in node.graph.out_edges(node.id, data=True, keys=True)
                 if 'out' in params and params['out'] == out_port]
        edge_attrs = edges[0][3]
        [self.graph.remove_edge(u, v, key=key) for u, v, key, params in edges]
        if attrs is None:
            attrs = dict()
        new_node = self.add_node(attrs)
        self.graph.add_edge(node.id, new_node.id, **edge_attrs)
        return new_node

    def create_node(self, inputs: list = None, attrs: dict = None, edge_attrs: dict = None):
        # TODO pass also edge attributes to copy to newly created edges
        # TODO attrs should be matched with attrs()
        if inputs is not None:
            inputs = [Op.extract_port(inp) for inp in inputs]
        else:
            inputs = []
        if attrs is None:
            attrs = dict()
        new_node = self.add_node(attrs)
        # Missed careful handling of debug information
        for i, inp in enumerate(inputs):
            edge_attr = {'in': i, 'out': inp[1],
                         'in_attrs': ['in'],
                         'out_attrs': ['out'],
                         'data_attrs': []} if not inp[0].has_valid('kind') or inp[0].kind == 'op' \
                else {'in': i, 'in_attrs': ['in']}
            if edge_attrs is not None:
                edge_attr.update(edge_attrs)
            self.graph.add_edge(inp[0].id, new_node.id, **edge_attr)
        return new_node

    def create_node_with_data(self, inputs: list = None, attrs: dict = None,
                              data_nodes: [Node, np.ndarray, list] = None):
        """
        Creates a new node with given inputs and attrs and also creates data node that
        holds the op output value. Inputs should be data nodes (not op nodes).
        Work for ops with a single output port only.
        """
        if inputs is None:
            inputs = []
        if attrs is None:
            attrs = {}
        # No need to extract port, because input node should be a data node,
        # so there is no choice.
        new_op_node = self.add_node(attrs)
        # TODO Preserve debug infor
        self.graph.add_edges_from([(input.id, new_op_node.id, {'in': i}) for i, input in enumerate(inputs)])
        # TODO: Extend to the case when multiple output ports
        old_data_value = [None]
        old_data_shape = [None]
        if data_nodes is None:
            data_node = unique_id(self.graph)
            self.graph.add_node(data_node, **add_attrs_props(
                dict(kind='data', precision="FP32", name=data_node, value=None, shape=None, data_type=None,
                     infer=None)))
            data_nodes = [Node(self.graph, data_node)]
        else:
            if type(data_nodes) not in [list, np.ndarray]:
                data_nodes = [data_nodes]
            old_data_value = [data_node.value.copy() if data_node.has_valid('value') else None for data_node in
                              data_nodes]
            old_data_shape = [data_node.shape.copy() if data_node.has_valid('shape') else None for data_node in
                              data_nodes]
        for id, data_node in enumerate(data_nodes):
            self.graph.add_edges_from([(new_op_node.id, data_node.id, {'out': id})])
        if new_op_node.has_valid('infer'):
            log.debug('Start running infer function for individual op node with attributes: {}'.format(
                new_op_node.graph.node[new_op_node.id]))
            new_op_node.infer(new_op_node)
            assert all(old_value is None for old_value in old_data_value) or all(
                [np.array_equal(old_data_value[id], data_node.value) for id, data_node in enumerate(data_nodes)])
            assert all(old_shape is None for old_shape in old_data_shape) or all(
                [np.array_equal(old_data_shape[id], data_node.shape) for id, data_node in enumerate(data_nodes)])
            for data_node in data_nodes:
                log.debug(
                    'Finished running infer function, data nodes attributes: {}'.format(
                        data_node.graph.node[data_node.id]))
        return data_nodes[0] if len(data_nodes) == 1 else data_nodes

    @staticmethod
    def create_data_node(graph: nx.MultiDiGraph, op_node: Node, attrs: dict = None):
        assert op_node is not None and op_node.kind == 'op'
        assert len(op_node.out_nodes()) == 0
        if attrs is None:
            attrs = {}

        data_node = unique_id(graph, op_node.id)
        defaul_attrs = dict(kind='data', precision="FP32", name=data_node, value=None, shape=None, data_type=None,
                            infer=None)
        defaul_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(defaul_attrs))
        data_node = Node(graph, data_node)
        graph.add_edges_from([(op_node.id, data_node.id, {'out': 0})])
        return data_node

    @staticmethod
    def _create_data_node(graph: nx.MultiDiGraph, name: str, attrs: dict = None):
        if attrs is None:
            attrs = {}

        data_node = unique_id(graph, name)
        defaul_attrs = dict(kind='data', precision="FP32", name=data_node, value=None, shape=None, data_type=None,
                            infer=None)
        defaul_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(defaul_attrs))
        data_node = Node(graph, data_node)
        return data_node

    @staticmethod
    def create_input_data_node(graph: nx.MultiDiGraph, name: str, value: np.array, attrs: dict = {}):
        data_node = unique_id(graph, name)
        defaul_attrs = dict(kind='data', precision="FP32", name=data_node, value=np.array(value), shape=value.shape,
                            data_type=None, infer=None)
        defaul_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(defaul_attrs))
        return Node(graph, data_node)

    def update_node(self, node: Node, attrs: dict = None):
        """
        Updates/creates new attributes in node based on self.attrs and attrs.
        """
        new_attrs = {}
        new_attrs.update(self.attrs)
        if attrs:
            new_attrs.update(attrs)
        new_attrs = add_attrs_props(new_attrs)
        update_ie_fields(new_attrs)
        self.substitute_ie_attrs(new_attrs)
        for k, v in new_attrs.items():
            node[k] = v

    @classmethod
    def update_node_stat(cls, node: Node, attrs: dict = None):
        if attrs is None:
            attrs = dict()
        op = cls(node.graph, attrs)
        op.update_node(node)

    def supported_attrs(self):
        """
        Attributes that user should/can set for the operation
        """
        return []

    def backend_attrs(self):
        """
        Attributes that will be translated to back-end IR
        """
        return self.supported_attrs()

    @staticmethod
    def get_op_class_by_name(name: str):
        return __class__.registered_ops[name]

    @classmethod
    def class_type(cls):
        return class_registration.ClassType.OP

    @staticmethod
    def expand_node_shape(node: Node, dims_to_add):
        if node is None or not node.has_valid('value'):
            return
        for idx in range(dims_to_add):
            node.value = np.expand_dims(node.value, axis=-1)
        node.shape = np.array(node.value.shape)
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy
import logging as log
from collections import namedtuple

import networkx as nx
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, strict_compare_tensors
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import add_attrs_props, update_ie_fields
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.utils import class_registration
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.runtime_info import RTInfo


class Op(object):
    registered_ops = {}
    registered_cls = []
    # Add the derived class to excluded_classes if one should not be registered in registered_ops
    excluded_classes = []

    def __init__(self, graph: Graph, attrs1: dict = None, attrs2: dict = None):
        self.graph = graph
        try:
            self.ir_version = graph.graph['ir_version']
        except:
            self.ir_version = None

        self.attrs = {
            'kind': 'op',
            'rt_info': RTInfo()
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
        id = self.graph.unique_id(id_prefix)
        new_attrs['name'] = id
        new_attrs = add_attrs_props(new_attrs)
        update_ie_fields(new_attrs, self.ir_version)
        self.substitute_ie_attrs(new_attrs)
        self.graph.add_node(id, **new_attrs)

        node = Node(self.graph, id)
        return node

    def substitute_ie_attrs(self, new_attrs: dict):
        """
        Replace standard list of attribute in layer/data by attributes
        delivered by backend_attrs
        """
        backend_attrs_mapping = {
            None: self.backend_attrs,
            10: self.backend_attrs,
            11: self.backend_attrs,
        }

        if self.ir_version not in backend_attrs_mapping.keys():
            raise Error("Unrecognized IR version was specified: {}".format(self.ir_version))

        new_attrs.update({
            'IE': [(
                'layer',
                [('id', lambda node: node.node), 'name', 'type', 'version'],
                [
                    ('data', backend_attrs_mapping[self.ir_version]() + self.default_backend_attrs, []),
                    '@runtime_info',
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
        # 'data' nodes do not have 'out' edge attribute but always has one output
        out_ids = [attr['out'] for _, __, attr in node.graph.out_edges(node.id, data=True) if 'out' in attr]
        if len(set(out_ids)) > 1 and not isinstance(node_port, tuple):
            raise Error('Node {} has more than one outputs. Provide output port explicitly. '.format(node.name))
        return node, port

    def create_node_on_port(self, node: Node, out_port: int, attrs: dict = None, edge_attrs: dict = None):
        """
        Removes an edge, that is connected to nodes out_port. Creates new_node with attrs attributes and
        connects it to node by edge that stores the same information as cutted edge.
        :param node: Input node, to cut the edge from
        :param out_port: output port of edge to cut
        :param attrs: attributes of new node
        :param edge_attrs: attributes to be changed/added to new edge
        :return: Node instance of created new_node
        """
        if edge_attrs is None:
            edge_attrs = {'in': 0}
        prev_edge_attrs = copy.deepcopy(node.out_edge(out_port))
        prev_edge_attrs.update(edge_attrs)
        new_edge_attrs = prev_edge_attrs
        if attrs is None:
            attrs = dict()
        new_node = self.add_node(attrs)
        self.graph.add_edge(node.id, new_node.id, **new_edge_attrs)
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
        for i, inp in enumerate(inputs):
            edge_attr = {'in': i, 'out': inp[1],
                         'in_attrs': ['in', 'permutation'],
                         'out_attrs': ['out', 'permutation'],
                         'data_attrs': []} if not inp[0].has_valid('kind') or inp[0].kind == 'op' \
                else {'in': i, 'in_attrs': ['in', 'permutation']}

            # handling of debug information
            if inp[0].has_port('out', inp[1]):
                debug_info = inp[0].out_port(inp[1]).get_tensor_debug_info()
                if debug_info is not None and len(debug_info) > 0:
                    edge_attr.update({'fw_tensor_debug_info': debug_info})
                    edge_attr['data_attrs'].append('fw_tensor_debug_info')

            if edge_attrs is not None:
                edge_attr.update(edge_attrs)
            new_node.add_input_port(i, skip_if_exist=True)
            inp[0].add_output_port(inp[1], skip_if_exist=True)
            self.graph.add_edge(inp[0].id, new_node.id, **edge_attr)
        return new_node

    def create_node_with_data(self, inputs: list = None, attrs: dict = None,
                              data_nodes: [Node, np.ndarray, list] = None, edge_attrs: list = None):
        """
        Creates a new node with given inputs and attrs and also creates data node that
        holds the op output value. Inputs should be data nodes (not op nodes).
        Work for ops with a single output port only.
        Edge attributes in edge_attrs go in order of items in 'inputs'
        """
        if inputs is None:
            inputs = []
        if attrs is None:
            attrs = {}
        # No need to extract port, because input node should be a data node,
        # so there is no choice.
        new_op_node = self.add_node(attrs)

        # TODO Preserve debug information
        inputs_with_edge_attrs = []
        for i, inp in enumerate(inputs):
            if inp is None:
                continue
            edge_attr = {'in': i}
            if edge_attrs is not None and i < len(edge_attrs):
                edge_attr.update(edge_attrs[i])
            inputs_with_edge_attrs.append((inp.id, new_op_node.id, edge_attr))
            new_op_node.add_input_port(i, skip_if_exist=True)

        self.graph.add_edges_from(inputs_with_edge_attrs)
        
        # TODO: Extend to the case when multiple output ports
        old_data_value = [None]
        old_data_shape = [None]
        if data_nodes is None:
            data_node = self.graph.unique_id()
            self.graph.add_node(data_node, **add_attrs_props(
                dict(kind='data', name=data_node, value=None, shape=None, data_type=None, infer=None)))
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
            if log.getLogger().isEnabledFor(log.DEBUG):
                log.debug('Start running infer function for individual op node with attributes: {}'
                          ''.format(str(new_op_node)))
            new_op_node.infer(new_op_node)
            if new_op_node.has('nchw_layout'):
                for out_node in new_op_node.out_nodes().values():
                    out_node['nchw_layout'] = new_op_node.nchw_layout
            assert all(old_value is None for old_value in old_data_value) or all(
                [strict_compare_tensors(old_data_value[id], data_node.value)
                 for id, data_node in enumerate(data_nodes)])
            assert all(old_shape is None for old_shape in old_data_shape) or all(
                [strict_compare_tensors(old_data_shape[id], data_node.shape)
                 for id, data_node in enumerate(data_nodes)]), \
                "After re-inference of {} node, old and new shapes do not match. Old shapes: {}, new shapes: {}." \
                "".format(new_op_node.soft_get('name'), [old_data_shape[id] for id in range(len(data_nodes))],
                          [data_node.shape for data_node in data_nodes])
            for data_node in data_nodes:
                if log.getLogger().isEnabledFor(log.DEBUG):
                    log.debug(
                        'Finished running infer function, data nodes attributes: {}'.format(data_node))
        return data_nodes[0] if len(data_nodes) == 1 else data_nodes

    @staticmethod
    def create_data_node(graph: Graph, op_node: Node, attrs: dict = None, edge_attrs: dict = None, out_port=0):
        assert op_node is not None and op_node.kind == 'op'
        assert out_port not in op_node.out_nodes()

        if attrs is None:
            attrs = {}

        data_node = graph.unique_id(op_node.id)
        default_attrs = dict(kind='data', name=data_node, value=None, shape=None, data_type=None, infer=None)
        default_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(default_attrs))
        data_node = Node(graph, data_node)
        if edge_attrs is not None:
            graph.add_edges_from([(op_node.id, data_node.id, {'out': out_port, **edge_attrs})])
        else:
            graph.add_edges_from([(op_node.id, data_node.id, {'out': out_port})])
        return data_node

    @staticmethod
    def _create_data_node(graph: Graph, name: str, attrs: dict = None):
        if attrs is None:
            attrs = {}

        data_node = graph.unique_id(name)
        default_attrs = dict(kind='data', name=data_node, value=None, shape=None, data_type=None, infer=None)
        default_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(default_attrs))
        data_node = Node(graph, data_node)
        return data_node

    @staticmethod
    def create_input_data_node(graph: Graph, name: str, value: np.array, attrs: dict = None):
        if attrs is None:
            attrs = {}
        data_node = graph.unique_id(name)
        default_attrs = dict(kind='data', name=data_node, value=mo_array(value), shape=mo_array(value.shape),
                             data_type=None, infer=None)
        default_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(default_attrs))
        return Node(graph, data_node)

    @staticmethod
    def create_and_connect_input_data_node(graph: Graph, op_node: Node, attrs: dict = None, edge_attrs: dict = None):
        assert op_node is not None and op_node.kind == 'op'
        if attrs is None:
            attrs = {}
        if edge_attrs is None:
            edge_attrs = {}

        data_node = graph.unique_id(op_node.id)
        default_attrs = dict(kind='data', name=data_node, value=None, shape=None, data_type=None, infer=None)
        default_attrs.update(attrs)
        graph.add_node(data_node, **add_attrs_props(default_attrs))
        data_node = Node(graph, data_node)
        op_node.add_input_port(edge_attrs['in'], skip_if_exist=True)
        graph.add_edges_from([(data_node.id, op_node.id, edge_attrs)])
        return data_node

    def update_node(self, node: Node, attrs: dict = None):
        """
        Updates/creates new attributes in node based on self.attrs and attrs.
        """
        new_attrs = {}
        new_attrs.update(self.attrs)
        if attrs:
            new_attrs.update(attrs)
        new_attrs = add_attrs_props(new_attrs)
        update_ie_fields(new_attrs, self.ir_version)
        self.substitute_ie_attrs(new_attrs)
        for k, v in new_attrs.items():
            node[k] = v
        node.update_node()

    def get_opset(self):
        """
        Gets the operation set version where the operation was introduced.
        If the version is not defined then consider it an extension
        :return: the string with the opset name
        """
        return self.attrs.get('version', 'extension')

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
        node.shape = mo_array(node.value.shape)

    @staticmethod
    def normalize_outputs(node: Node):
        if node.has_valid('out_ports_count') and len(node.out_edges()) < node.out_ports_count:
            from openvino.tools.mo.ops.result import Result    # Import is here to avoid circular import error
            for p in range(node.out_ports_count):
                if p not in node.out_ports():
                    node.add_output_port(p)
                if node.out_port(p).disconnected():
                    res_node = Result(node.graph, {'name': node.name + '/Fake_output_{}/'.format(p),
                                                   'keep_output_port': True}).create_node()
                    node.out_port(p).connect(res_node.in_port(0))


class PermuteAttrs:
    Permutation = namedtuple('Permutation', ['perm', 'inv'])
    Attr = namedtuple('Attr', ['name', 'port', 'func'])

    common_permutation = lambda node, permutation, attr: node[attr][permutation.perm]
    slice_permutation = lambda node, permutation, attr: node[attr][  # doesn't depend from permutation variable
        PermuteAttrs.get_nhwc_to_nchw_permutation(len(node[attr])).perm]
    common_permutation_inv = lambda node, permutation, attr: permutation.inv[node[attr]]

    # List of default permutations
    common_attrs_permutation = {
            'dim': common_permutation,
            'pad': common_permutation,
            'pads': common_permutation,
            'shape': common_permutation,
            'order': lambda node, permutation, attr: permutation.inv[node[attr][permutation.perm]],
            'stride': common_permutation,
            'window': common_permutation,
            'dilation': common_permutation,
            'kernel_shape': common_permutation,
            'output_shape': common_permutation,
            'begin_mask': slice_permutation,
            'end_mask': slice_permutation,
            'shrink_axis_mask': slice_permutation,
            'new_axis_mask': slice_permutation,
            'ellipsis_mask': slice_permutation,
            'axes': common_permutation_inv,
            'axis': common_permutation_inv,
            'seq_axis': common_permutation_inv,
            'batch_axis': common_permutation_inv,
            'batch_dims': common_permutation_inv,
            'channel_dims': common_permutation_inv,
            'spatial_dims': common_permutation_inv,

            'input_channel_dim': common_permutation_inv,
            'output_channel_dim': common_permutation_inv,
            'kernel_spatial_idx': common_permutation_inv,
            'input_feature_channel': common_permutation_inv,
            'output_feature_channel': common_permutation_inv,
    }

    @staticmethod
    def __attr(name, port, func=None):
        if func is None:
            if name in PermuteAttrs.common_attrs_permutation:
                func = PermuteAttrs.common_attrs_permutation[name]
            else:
                raise Error('Attr {} is missing in PermuteAttrs.common_attrs_permutation. Please update '
                            'common_attrs_permutation with permutation for your attribute!'.format(name))

        if len(port.split(':')) != 2 or port.split(':')[0] not in ['input', 'output']:
            raise Error("Attribute port {} for {} wasn't set correctly!".format(port, name))

        return PermuteAttrs.Attr(name=name, port=port, func=func)

    def __init__(self):
        self.attrs = {}

    def update_attrs(self, attrs):
        for attr in attrs:
            if not isinstance(attr, tuple) or len(attr) not in [2, 3]:
                raise Error('attr object must be a tuple: (attribute_name, port) or (attribute_name, port, func)')
            self.attrs.update({attr[0]: self.__attr(*attr)})
        return self

    def permute_attrs(self, node):
        # This function applies permutation for given node
        for attr in self.attrs.keys():
            name, port, func = self.attrs[attr]
            node_type, port = port.split(':')
            port = int(port)
            node_with_permutation = node.in_node(port) if node_type == 'input' else node.out_node(port)

            if node_with_permutation.has_valid('permutation'):
                permutation = node_with_permutation.permutation
                if isinstance(permutation, type(lambda: 0)):
                    node[name] = func(node, permutation(node), name)
                else:
                    node[name] = func(node, permutation, name)

    @staticmethod
    def create_permute_attrs(node, attrs=None):
        # Create permute_attrs if not exists
        if not node.has_valid('permute_attrs'):
            node['permute_attrs'] = PermuteAttrs()
        node['permute_attrs'].update_attrs(attrs)

    @staticmethod
    def set_permutation(node1, node2, permutation, override=False):
        # This function creates permutation on edge between node1->node2
        edge_attrs = node1.graph.get_edge_data(node1.id, node2.id)[0]
        if 'permutation' not in edge_attrs or override:
            nx.set_edge_attributes(G=node1.graph, values={(node1.id, node2.id, 0): permutation}, name='permutation')
        else:
            # If permutation exists we check that given and already set permutations are equal
            if (edge_attrs['permutation'] is None and permutation is not None) or \
                    not np.array_equal(edge_attrs['permutation'], permutation):
                raise Error('Permutation already exists in edge between {} and {}'.format(node1.id, node2.id))

    @staticmethod
    def get_inverse_permutation(perm):
        inv = [0] * len(perm)
        # Create reverse permutation
        for index, pos in enumerate(perm):
            inv[pos] = index
        return inv

    @staticmethod
    def get_nhwc_to_nchw_permutation(dims_number: int):
        # This function returns permutation from NHWC to NCHW for given dims number
        if dims_number != 3:
            perm = [0, dims_number - 1, *[x for x in range(1, dims_number - 1)]] if dims_number > 1 else \
                [x for x in range(dims_number)]
        else:
            # Exclude 3D shapes from permutation process: identity permutation
            perm = list(range(0, dims_number))
        inv = PermuteAttrs.get_inverse_permutation(perm)
        return PermuteAttrs.Permutation(perm=int64_array(perm), inv=int64_array(inv))

    @staticmethod
    def get_nchw_to_nhwc_permutation(dims_number: int):
        # This function returns permutation from NCHW to NHWC for given dims number
        if dims_number != 3:
            perm = [0, *[x for x in range(2, dims_number)], 1] if dims_number > 1 else [x for x in range(dims_number)]
        else:
            # Exclude 3D shapes from permutation process: identity permutation
            perm = list(range(0, dims_number))
        inv = PermuteAttrs.get_inverse_permutation(perm)
        return PermuteAttrs.Permutation(perm=int64_array(perm), inv=int64_array(inv))

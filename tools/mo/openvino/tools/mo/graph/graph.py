# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import collections
import logging as log
from copy import deepcopy
from typing import List

import networkx as nx
import numpy as np

from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.passes.eliminate import mark_output_reachable_nodes, shape_inference, mark_undead_nodes, \
    mark_const_producer_nodes, eliminate_dead_nodes, add_constant_operations
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg, deprecated_api, shrink_str_value


def dict_to_ordered_dict(d: dict, func=lambda t: t):
    return collections.OrderedDict(sorted(d.items(), key=lambda t: func(t[0])))


class Node:
    def __init__(self, graph, node: str):
        assert node in graph, "Attempt to access node {} that not in graph".format(node)

        super(Node, self).__setattr__('graph', graph)
        super(Node, self).__setattr__('node', node)  # obsolete
        super(Node, self).__setattr__('id', node)

    def __str__(self, max_length: int = 100):
        node_dict = self.graph.node[self.id]
        print_dict = {k: v if k != 'value' else shrink_str_value(v, max_symbols=max_length) for k, v in
                      node_dict.items()}
        return str(print_dict)

    def __setattr__(self, k, v):
        # you can assign only existing attributes
        attrs = self.graph.node[self.node]
        if not k in attrs:
            raise AttributeError("Attribute {} missing in {} node".format(k, self.name))
        if k == 'version' and attrs.get(k, v) != v:
            raise AttributeError("Attribute 'version' cannot be updated in {} node".format(self.name))

        attrs[k] = v

    def __getattr__(self, k):
        return self.graph.node[self.node][k]

    def __getitem__(self, k):
        return self.graph.node[self.node][k]

    def __setitem__(self, k, v):
        if k == 'version' and self.graph.node[self.node].get(k, v) != v:
            raise AttributeError("Attribute 'version' cannot be updated in {} node".format(self.name))
        self.graph.node[self.node][k] = v

    def __contains__(self, k):
        return self.has(k)

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.graph == other.graph and
                self.id == other.id
        )

    def __hash__(self):
        return hash((self.graph, self.id))

    def __delitem__(self, k):
        del self.graph.node[self.node][k]

    def add_input_port(self, idx, skip_if_exist=False, **kwargs):
        if not self.has_valid('_in_ports'):
            Node(self.graph, self.id)['_in_ports'] = {}
        control_flow = kwargs['control_flow'] if kwargs.get('control_flow') is not None else False
        if skip_if_exist is False and idx in self.in_ports(control_flow=control_flow):
            raise Error("Input port with {} index already exists for {} node.".format(idx, self.name))
        self._in_ports.update({idx: kwargs})

    def delete_input_ports(self, idx_set, skip_if_absent=False):
        if len(idx_set) == 0:
            return  # there is nothing to delete
        for idx in idx_set:
            self.delete_input_port(idx, skip_if_absent)

    def delete_input_port(self, idx, skip_if_absent=False):
        if not self.has_valid('_in_ports'):
            raise Error(
                'Cannot removed ports with indices {} from node {} because node doesn\'t '
                'have _in_ports attribute'.format(idx, self.soft_get('name')))
        # no handling of control flow edges -- TODO
        control_flow = False
        if not skip_if_absent and idx not in self.in_ports(control_flow=control_flow):
            raise Error("Input port with index {} doesn't exist in node {}.".format(idx, self.soft_get('name')))
        if not self.in_port(idx).disconnected():
            self.in_port(idx).disconnect()
        del self._in_ports[idx]
        # update in_ports_count for consistency but it is unlikely have any effect somewhere in the code
        self['in_ports_count'] = len(self._in_ports)

    def delete_output_port(self, idx, skip_if_absent=False):
        if not self.has_valid('_out_ports'):
            raise Error(
                'Cannot removed ports with indices {} from node {} because node doesn\'t '
                'have _out_ports attribute'.format(idx, self.soft_get('name')))
        # no handling of control flow edges -- TODO
        control_flow = False
        if not skip_if_absent and idx not in self.out_ports(control_flow=control_flow):
            raise Error("Output port with index {} doesn't exist in node {}.".format(idx, self.soft_get('name')))
        if not self.out_port(idx).disconnected():
            self.out_port(idx).disconnect()
        del self._out_ports[idx]
        # update in_ports_count for consistency but it is unlikely have any effect somewhere in the code
        self['out_ports_count'] = len(self._out_ports)

    def add_output_port(self, idx, skip_if_exist=False, **kwargs):
        if not self.has_valid('_out_ports'):
            Node(self.graph, self.id)['_out_ports'] = {}
        control_flow = kwargs['control_flow'] if kwargs.get('control_flow') is not None else False
        if skip_if_exist is False and idx in self.out_ports(control_flow=control_flow):
            raise Error("Output port with {} index already exists for {} node.".format(idx, self.name))
        self._out_ports.update({idx: kwargs})

    def add_sequence_of_ports(self, type: str, rng):
        assert type in ['in', 'out']
        for idx in rng:
            if type == 'in':
                self.add_input_port(idx, skip_if_exist=True)
            if type == 'out':
                self.add_output_port(idx, skip_if_exist=True)

    def in_port(self, idx=None, control_flow=False) -> Port:
        if not self.has_valid('_in_ports'):
            raise Error("Operation {} {} has no _in_ports attribute", self.op, self.name)
        if idx not in self._in_ports:
            raise Error("Input port with index {} is not in node {}".format(idx, self.name))
        if not control_flow and 'control_flow' in self._in_ports[idx] and self._in_ports[idx]['control_flow']:
            raise Error("Attempt to access control flow port when it's prohibited for node {}".format(self.name))
        return Port(node=self, idx=idx, type='in', **self._in_ports[idx])

    def in_ports(self, control_flow=False):
        if not self.has_valid('_in_ports'):
            raise Error("Operation {} {} has no _in_ports attribute", self.op, self.name)
        ports = {}
        for idx in self._in_ports:
            if control_flow or 'control_flow' not in self._in_ports[idx] or not self._in_ports[idx]['control_flow']:
                ports.update({idx: self.in_port(idx, control_flow=control_flow)})
        return dict_to_ordered_dict(ports, func=lambda t: int(str(t).replace('control_flow_', '')))

    def out_port(self, idx=None, control_flow=False) -> Port:
        if not self.has_valid('_out_ports'):
            raise Error("Operation {} {} has no _out_ports attribute", self.op, self.name)
        if idx not in self._out_ports:
            raise Error("Output port with index {} is not in node {}".format(idx, self.name))
        if not control_flow and 'control_flow' in self._out_ports[idx] and self._out_ports[idx]['control_flow']:
            raise Error("Attempt to access control flow port when it's prohibited for node {}".format(self.name))
        return Port(node=self, idx=idx, type='out', **self._out_ports[idx])

    def out_ports(self, control_flow=False):
        if not self.has_valid('_out_ports'):
            raise Error("Operation {} {} has no _out_ports attribute", self.op, self.name)
        ports = {}
        for idx in self._out_ports:
            if control_flow or 'control_flow' not in self._out_ports[idx] or not self._out_ports[idx]['control_flow']:
                ports.update({idx: self.out_port(idx, control_flow=control_flow)})
        return dict_to_ordered_dict(ports, func=lambda t: int(str(t).replace('control_flow_', '')))

    def has_port(self, port_type, idx, control_flow=False):
        assert port_type in ['in', 'out'], "Invalid usage of has_port method"

        if port_type == 'in':
            return self.has_valid('_in_ports') and idx in self.in_ports(control_flow=control_flow)
        else:
            return self.has_valid('_out_ports') and idx in self.out_ports(control_flow=control_flow)

    def is_in_port_connected(self, idx, control_flow=False):
        return self.has_port('in', idx, control_flow) and not self.in_port(idx, control_flow).disconnected()

    def is_out_port_connected(self, idx, control_flow=False):
        return self.has_port('out', idx, control_flow) and not self.out_port(idx, control_flow).disconnected()

    def attrs(self):
        return self.graph.node[self.node]

    def has(self, k):
        return k in self.graph.node[self.node]

    def has_valid(self, k):
        return self.has(k) and not self.graph.node[self.node][k] is None

    def has_and_set(self, k):
        return self.has_valid(k) and self[k]

    def in_nodes_edges(self, control_flow: bool = False):
        return dict_to_ordered_dict({x[1]['in']: (Node(self.graph, x[0]), x[1]) for x in
                                     self.get_inputs(control_flow=control_flow)},
                                    func=lambda t: int(str(t).replace('control_flow_', '')))

    def in_nodes(self, control_flow: bool = False):
        if self.kind == 'op':
            return dict_to_ordered_dict({x[1]['in']: Node(self.graph, x[0]) for x in
                                         self.get_inputs(control_flow=control_flow)},
                                        func=lambda t: int(str(t).replace('control_flow_', '')))
        elif self.kind == 'data':
            return [Node(self.graph, n) for n, d in self.get_inputs(control_flow=control_flow)]

    def in_node(self, key=0, control_flow: bool = False):
        return self.in_nodes(control_flow=control_flow)[key]

    def in_edges(self, control_flow: bool = False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return dict_to_ordered_dict({x[1]['in']: x[1] for x in self.get_inputs(control_flow=control_flow)},
                                        func=lambda t: int(str(t).replace('control_flow_', '')))
        elif self.kind == 'data':
            return [d for n, d in self.get_inputs(control_flow=control_flow)]

    def out_nodes_edges(self, control_flow: bool = False):
        return dict_to_ordered_dict({x[1]['out']: (Node(self.graph, x[0]), x[1]) for x in
                                     self.get_outputs(control_flow=control_flow)},
                                    func=lambda t: int(str(t).replace('control_flow_', '')))

    def out_nodes(self, control_flow: bool = False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return dict_to_ordered_dict({x[1]['out']: Node(self.graph, x[0]) for x in
                                         self.get_outputs(control_flow=control_flow)},
                                        func=lambda t: int(str(t).replace('control_flow_', '')))
        elif self.kind == 'data':
            return [Node(self.graph, n) for n, d in self.get_outputs(control_flow=control_flow)]

    def out_edges(self, control_flow: bool = False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return dict_to_ordered_dict({x[1]['out']: x[1] for x in self.get_outputs(control_flow=control_flow)},
                                        func=lambda t: int(str(t).replace('control_flow_', '')))
        elif self.kind == 'data':
            return [d for n, d in self.get_outputs(control_flow=control_flow)]

    def out_node(self, key=0, control_flow: bool = False):
        return self.out_nodes(control_flow=control_flow)[key]

    def in_edge(self, key=0, control_flow: bool = False):
        return self.in_edges(control_flow=control_flow)[key]

    def out_edge(self, key=0, control_flow: bool = False):
        return self.out_edges(control_flow=control_flow)[key]

    def get_attrs(self):
        return self.graph.node[self.node]

    def get_inputs(self, edge_attr: dict = None, control_flow: bool = False):
        if edge_attr is None:
            edge_attr = {}
        in_edges = self.graph.in_edges(self.id, data=True)
        if not control_flow:
            in_edges = [(u, v, d) for u, v, d in in_edges if 'control_flow_edge' not in d or not d['control_flow_edge']]
        return [(u, d) for u, v, d in in_edges if all([attr in d and d[attr] == edge_attr[attr] for attr in edge_attr])]

    def get_outputs(self, edge_attr: dict = None, control_flow: bool = False):
        if edge_attr is None:
            edge_attr = {}
        out_edges = self.graph.out_edges(self.id, data=True)
        if not control_flow:
            out_edges = [(u, v, d) for u, v, d in out_edges if
                         'control_flow_edge' not in d or not d['control_flow_edge']]
        return [(v, d) for u, v, d in out_edges if
                all([attr in d and d[attr] == edge_attr[attr] for attr in edge_attr])]

    def get_sorted_inputs(self, control_flow: bool = False):
        return sorted([x for x in self.get_inputs(control_flow=control_flow) if 'in' in x[1]],
                      key=lambda x: x[1]['in'])

    def get_sorted_outputs(self, control_flow: bool = False):
        return sorted([x for x in self.get_outputs(control_flow=control_flow) if 'out' in x[1]],
                      key=lambda x: x[1]['out'])

    def soft_get(self, k, default='<UNKNOWN>'):
        return self[k] if self.has_valid(k) else default

    def edges(self, attrs: dict = None):
        """ Get a single edge with specified set of attributes.

            If none or multiple edges satisfies this criteria, exception is raised
            Edge is represented as tuple (u, v, d), where u is source node,
            v is destination node and d is edge attributes.
        """
        edges = list(self.graph.in_edges([self.id], data=True)) + list(self.graph.out_edges([self.id], data=True))
        return [(u, v, d) for u, v, d in edges if dict_includes(d, attrs)]

    def edge(self, attrs: dict = None):
        """ Get a single edge with specified set of attributes.

            If none or multiple edges satisfies this criteria, exception is raised
            Edge is represented as tuple (u, v, d), where u is source node,
            v is destination node and d is edge attributes.
        """
        edges = self.edges(attrs)
        assert len(edges) == 1, 'edges: {}, required attributes: {}'.format(edges, attrs)
        return edges[0]

    def copy_node(self, new_attrs: dict = None, dst_graph=None):
        ''' Copies node with all attributes (optionally updated) within the same graph or to different graph.'''
        if new_attrs is None:
            new_attrs = {}
        if dst_graph is None:
            dst_graph = self.graph

        attrs = deepcopy(self.attrs())
        new_id = dst_graph.unique_id(attrs['name']) if 'name' in attrs else dst_graph.unique_id()
        attrs['name'] = new_id
        attrs.update(new_attrs)
        dst_graph.add_node(new_id, **attrs)
        return Node(dst_graph, new_id)

    def insert_node_with_data_before(self, inp, new_op_class: callable, op_before_params: dict = None,
                                     infer_current: bool = False, additional_inputs: list = None):
        """
        Inserts operation node with op_before_params and data node before current operation

        :param inp: input data node of current node
        :param new_op_class: class of operation that will be inserted before current operation node
        :param op_before_params: parameters to be added to operation that will be inserted before current operation

        Before calling:
        [...] -> inp -> Cur_Op -> Cur_Data -> [...]

        After calling:
        [...] -> inp -> New_Op_bef -> New_Data_bef -> Cur_Op -> Cur_Data -> [...]
                    [op_before_params]
        """
        graph = self.graph
        node = Node(graph, self.node)
        cls_name = new_op_class.op
        op_before_params = {} if op_before_params is None else op_before_params

        # operating with input
        new_op_before = new_op_class(graph, op_before_params)
        edge_attrs = deepcopy(graph.get_edge_data(inp.id, node.id)[0])
        graph.remove_edge(inp.id, node.id)
        # form a list of input nodes for a new op node combining new_out and additional_inputs
        inputs = [inp] + (additional_inputs if additional_inputs else [])
        new_inp = new_op_before.create_node_with_data(inputs, {'name': node.name + cls_name + '/Before'})
        graph.add_edge(new_inp.id, node.id, **edge_attrs)
        if infer_current:
            node.infer(node)

    def insert_node_with_data_after(self, out, new_op_class: callable, op_after_params: dict = None,
                                    additional_inputs: list = None):
        """
        Inserts operation node with op_after_params and data node after current operation

        :param out: output data node of current node
        :param new_op_class: class of operation that will be inserted after current operation node
        :param op_after_params:  parameters to be added to operation that will be inserted after current operation
        :param additional_inputs:  other parameters for a new operation node in addition to one that is created
            at the 'out' placed; new nodes are added after 0-th input

            TODO Allow indexing for input parameters as well as for 'out' data node to explicitly
                specify ports that are connected to.

        Before calling:
        [...] -> Cur_Op -> Cur_Data -> [...]

        After calling:
        [...] -> Cur_Op -> Cur_Data -> New_Op_aft -> New_Data_aft(==out) -> [...]
                                   [op_after_params]
        """
        # we import it here because Op imports Node and unique_id from this file
        from openvino.tools.mo.ops.op import Op

        graph = self.graph
        node = Node(graph, self.node)
        cls_name = new_op_class.op
        op_after_params = {} if op_after_params is None else op_after_params

        new_op_after = new_op_class(graph, op_after_params)
        graph.remove_edge(node.id, out.id)
        new_out = Op.create_data_node(graph, node)
        node.infer(node)
        # form a list of input nodes for a new op node combining new_out and additional_inputs
        inputs = [new_out] + (additional_inputs if additional_inputs else [])
        new_op_after.create_node_with_data(inputs, {'name': node.name + cls_name + '/After'}, data_nodes=out)

    def bracket_with_different_nodes_with_data(self, inp, out, new_op_class_before: callable,
                                               new_op_class_after: callable,
                                               op_before_params: dict = None, op_after_params: dict = None):
        """
        Inserts one operation node with op_before_params and data node before current operation node and
        inserts one operation node with op_after_params and data node after current operation node
        :param inp: input data node of self.node node
        :param out: output data node of self.node node
        :param new_op_class_before: class of operation that will be inserted before current operation node
        :param new_op_class_after: class of operation that will be inserted after current operation node
        :param op_before_params: parameters to be added to operation that will be inserted before current operation
        :param op_after_params: parameters to be added to operation that will be inserted after current operation

        Before calling:
        [...] -> inp -> Cur_Op -> out -> [...]

        After calling:
        [...] -> inp -> New_Op_bef -> New_Data_bef -> Cur_Op -> Cur_Data -> New_Op_aft -> New_Data_aft(==out) -> [...]
                    [op_before_params]                                  [op_after_params]
        """
        op_before_params = {} if op_before_params is None else op_before_params
        op_after_params = {} if op_after_params is None else op_after_params
        self.insert_node_with_data_before(inp, new_op_class_before, op_before_params)
        self.insert_node_with_data_after(out, new_op_class_after, op_after_params)

    def bracket_op_with_another_op(self, inp, out, new_op_class: callable,
                                   op_before_params: dict = None, op_after_params: dict = None):
        """
        Covers current operation with two similar another ones of class new_op_class:
        :param inp: input data node of self.node node
        :param out: output data node of self.node node
        :param new_op_class: class of operation with which current operation will be covered
        :param op_before_params: parameters to be added to operation that will be inserted before current operation
        :param op_after_params: parameters to be added to operation that will be inserted after current operation

        Before calling:
        [...] -> inp -> Cur_Op -> out -> [...]

        After calling:
        [...] -> inp -> New_Op_bef -> New_Data_bef -> Cur_Op -> Cur_Data -> New_Op_aft -> New_Data_aft(==out) -> [...]
                    [op_before_params]                                  [op_after_params]
        """
        self.bracket_with_different_nodes_with_data(inp=inp, out=out,
                                                    new_op_class_before=new_op_class, new_op_class_after=new_op_class,
                                                    op_before_params=op_before_params, op_after_params=op_after_params)

    def insert_node_after(self, new_node, node_out_port: int = 0):
        """
        Insert node 'new_node' after output with index 'node_out_port' of the node 'node'. All consumers of node 'node'
        output with index 'node_out_port' will be changed to consume node 'new_node'.
        The function should be used when graph doesn't contain data nodes yet.
        :param node: node after which new node should be inserted.
        :param new_node: node to be inserted.
        :param node_out_port: the output index for the node 'node' to insert
        :return: None
        """
        assert self.graph is new_node.graph
        assert (len([name for name in self.graph.nodes() if Node(self.graph, name).soft_get('kind') == 'data']) == 0)

        graph = self.graph
        old_edges = list(graph.out_edges(self.id, data=True, keys=True))
        # create new edges first and then remove all old edges. This is needed for case when 'node' has several consumers
        # getting input from 'node_out_port'.
        # save tuple ("name of the destination edge", "edge key") to be removed
        node_name_and_edge_key = []
        for _, dst_name, edge_key, edge_attrs in old_edges:
            if edge_attrs['out'] == node_out_port:
                log.debug('Create edge from "{}" to "{}"'.format(new_node.name, dst_name))
                graph.create_edge(new_node, Node(graph, dst_name), 0, edge_attrs['in'])
                node_name_and_edge_key.append((dst_name, edge_key))
        for dst_name, edge_key in node_name_and_edge_key:
            log.debug('Remove edge from "{}" to "{}"'.format(self.id, dst_name))
            graph.remove_edge(self.id, dst_name, edge_key)
        graph.create_edge(self, new_node, node_out_port, 0, {})

    def insert_op_on_input_port(self, in_port_idx: int, new_op_class: callable, new_op_attrs: dict,
                                value: np.ndarray = None):
        """
        Inserts new operation of new_op_class on in_port_index input port with new_op_attrs
        Connects Const operation with value to 1 input port of new node if value was passed

        Returns new operation node
        """
        graph = self.graph
        name = self.soft_get('name', self.id)

        op_node = new_op_class(graph, new_op_attrs).create_node()

        assert self.has_port('in', in_port_idx), \
            'Node `{}` should have input port with idx `{}` but it does not'.format(name, in_port_idx)

        in_port_source = self.in_port(in_port_idx).get_source()
        self.in_port(in_port_idx).get_connection().set_source(op_node.out_port(0))
        op_node.in_port(0).connect(in_port_source)

        if value is not None:
            from openvino.tools.mo.ops.const import Const
            constant = Const(graph, {'value': value, 'name': op_node.name + '/value'}).create_node()
            op_node.in_port(1).connect(constant.out_port(0))

        return op_node

    def replace_node(self, new_node, new_node_out_port: int = None):
        """
        Replaces node 'old_node' with a node 'new_node' preserving edge attributes.
        :param old_node: node to be replaced.
        :param new_node: node to replace with.
        :return: None
        """
        assert self.graph is new_node.graph
        assert self.id != new_node.id, "New node and replaceable node are the same"
        graph = self.graph
        # save output edges and reconnect them to new node
        for _, dst_node_name, edge_attrs in graph.out_edges(self.id, data=True):
            new_edge_attrs = deepcopy(edge_attrs)
            if new_node_out_port is not None:
                assert 'out' not in edge_attrs or edge_attrs['out'] == 0, \
                    'replace_node function can replace old node with a single output port only if new_node_out_port is ' \
                    'specified'
                new_edge_attrs.update({'out': new_node_out_port})
            graph.add_edge(new_node.id, dst_node_name, **new_edge_attrs)

        # if the node for replace is output node then we propagate this attribute to a new node
        if len(self.out_nodes()) == 1 and self.out_node().has('op') and self.out_node().op == 'Result':
            graph.remove_node(self.out_node().id)
            add_opoutput(graph, new_node.id, 0, False)
        graph.remove_node(self.id)

    def input_ports_with(self, node):
        """
        Returns a list of integers that specify input ports that connected to a given node.
        :param node: node in the graph that is expected to appear at input port for self node
        :return: a list of integers with port indices that are connected to self node
        """
        return [i for i in range(len(self.in_nodes())) if self.in_node(i).id == node.id]

    def update_node(self):
        """
        Update internal node attributes. Currently it just add input/output ports.
        :return: None
        """
        in_ports_count = self.in_ports_count if self.has_valid('in_ports_count') else None
        out_ports_count = self.out_ports_count if self.has_valid('out_ports_count') else None

        if not self.has_valid('_in_ports'):
            Node(self.graph, self.id)['_in_ports'] = dict()
        if not self.has_valid('_out_ports'):
            Node(self.graph, self.id)['_out_ports'] = dict()

        if in_ports_count is not None:
            for idx in range(in_ports_count):
                if idx not in self._in_ports:
                    self.add_input_port(idx=idx)

        if out_ports_count is not None:
            for idx in range(out_ports_count):
                if idx not in self._out_ports:
                    self.add_output_port(idx=idx)

    def get_opset(self):
        """
        Gets the operation set version where the operation was introduced.
        If the version is not defined then consider it an extension
        :return: the string with the opset name
        """
        return self.soft_get('version', 'extension')


class Graph(nx.MultiDiGraph):
    def __init__(self, data=None, **attr):
        self.stage = None
        self.strict_mode = True
        super().__init__(data, **attr)

        if not hasattr(self, 'node'):
            self.node = self.nodes

    unique_id_count = 0
    op_names_statistic = collections.Counter()
    inputs_order = []
    outputs_order = []

    # SAFE API DESCRIPTION
    # all provided methods below are designed to be more safe and convenient
    # be careful while using other methods from nx.MultiDiGraph

    def add_node(self, node_for_adding, **attrs):
        # TODO: check required attrs for node
        super().add_node(node_for_adding, **attrs)
        node = Node(self, node_for_adding)
        node.update_node()

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):

        # TODO: turn on strict mode
        if self.strict_mode:
            unode = Node(self, u_for_edge)
            vnode = Node(self, v_for_edge)

            # Check that we connect Op->Op in front phase, and data->Op or Op->data in middle(back) phase
            # Also check that all necessary ports are exists
            message = "Attempt to connect {} to {}.".format(u_for_edge, v_for_edge)
            if self.stage == 'front':
                assert unode.kind == 'op' and vnode.kind == 'op', "{} Wrong add_adge usage! You can connect only two " \
                                                                  "operations in front phase".format(message)
                assert 'in' in attr and 'out' in attr, "Missing necessary attribute in or out when adding edge " \
                                                       "between {} and {}".format(u_for_edge, v_for_edge)
                is_control_flow = 'control_flow_edge' in attr and attr['control_flow_edge'] is True
                in_port = 'control_flow_{}'.format(attr['in']) if is_control_flow else attr['in']
                out_port = 'control_flow_{}'.format(attr['out']) if is_control_flow else attr['out']
                assert unode.has_port('out', out_port, control_flow=is_control_flow), \
                    "{} Missing out port ({}) in {} node".format(message, out_port, unode.soft_get('name', unode.id))
                assert vnode.has_port('in', in_port, control_flow=is_control_flow), \
                    "{} Missing in port ({}) in {} node".format(message, in_port, vnode.soft_get('name', vnode.id))
            elif self.stage in ['middle', 'back']:
                assert (unode.kind == 'data' and vnode.kind == 'op') or (unode.kind == 'op' and vnode.kind == 'data')
                if unode.kind == 'data' and vnode.kind == 'op':
                    assert 'in' in attr, "Attribute in is missing when adding edge to {}".format(v_for_edge)
                    assert vnode.has_port('in', attr['in']), "{} Node {} has no in port ({})" \
                                                             "".format(message, vnode.name, attr['in'])
                if unode.kind == 'op' and vnode.kind == 'data':
                    assert 'out' in attr, "Attribute out is missing when adding edge from {}".format(u_for_edge)
                    assert unode.has_port('out', attr['out']), "{} Node {} has no out port ({})" \
                                                               "".format(message, unode.name, attr['out'])

        return super().add_edge(u_for_edge, v_for_edge, key=key, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 4:
                u, v, key, dd = e
            elif ne == 3:
                u, v, dd = e
                key = None
            elif ne == 2:
                u, v = e
                dd = {}
                key = None
            else:
                raise Error("Edge tuple %s must be a 2-tuple, 3-tuple or 4-tuple." % (e,))
            ddd = attr.copy()
            ddd.update(dd)
            self.add_edge(u, v, key=key, **ddd)

    def remove_edge(self, u, v, key=None):
        return super().remove_edge(u, v, key=key)

    def erase_node(self, node: Node):
        """
        Erases node from the graph and reconnect edges from input node(s) to output node(s)
        Produces assertion error if the node being removed has multiple inputs or outputs.
        The function can be used in the front phase only (when there are no data nodes in the graph).
        :param node: Node to erase
        """
        node_id = node.id

        inputs = list(self.in_edges(node_id, data=True))
        outputs = list(self.out_edges(node_id, data=True))

        assert node.kind == 'op' and (len(node.out_nodes()) == 0 or list(node.out_nodes().values())[0].kind != 'data'), \
            "The function must be used before the partial infer when graph doesn't contain data nodes."
        assert len(node.out_nodes()) <= 1, "The node {} must produce just one output tensor".format(
            node.soft_get('name'))
        assert len(inputs) <= 1, "The node {} must have just one input".format(node.soft_get('name'))

        if len(outputs) == 0 and len(inputs) != 0:
            from openvino.tools.mo.front.extractor import add_output_ops
            input_ids = {input_node_id: {'port': {'out': [attrs['out']]}} for input_node_id, _, attrs in inputs}
            if node.has('op') and node.op == 'Result':
                add_output_ops(self, input_ids)

        if len(outputs) == 0 or len(inputs) == 0:
            self.remove_node(node_id)
            return

        input_node_id = inputs[0][0]
        for src, dst, attrs in outputs:
            self.remove_edge(src, dst)
            # update the 'out' attribute of the edge from the node being removed
            attrs['out'] = inputs[0][2]['out']
            self.add_edge(input_node_id, dst, **attrs)
        self.remove_node(node_id)

    def get_edge_data(self, u, v, key=None, default=None):
        return super().get_edge_data(u, v, key=key, default=default)

    def get_inputs_with_ports(self, match, pattern_edges, input_names_in_pattern):
        """
        Front replacements of multi-input nodes should specify output port to add_node-like functions
        This function is a helper to get such information out of matched nodes
        :param graph: graph to operate on
        :param match: dictionary returned by matching function
        :param pattern_edges: edges that are specified in pattern
        :param input_names_in_pattern: names of matched nodes as they were specified in pattern that should be in
        resulting list
        :return: list of tuples of node and output port
        """
        inputs = []
        for name in input_names_in_pattern:
            assert name in match, "node named {} not in match {}".format(name, match)
            src = match[name]
            dst = []
            for edge in pattern_edges:
                if edge[0] == name:
                    assert edge[1] in match, "name from pattern_edges {} not in match {}".format(edge[1], match)
                    dst.append(match[edge[1]])
            if len(dst) != 1:
                raise Error('Multiple output ports detected for node {} as {} in pattern'.format(match[name].id, name))
            dst = dst[0]
            out_port = self.get_edge_data(src.id, dst.id)[0]['out']
            inputs.append((src, out_port))
        return inputs

    def get_node_id_by_name(self, name: str):
        nodes = self.get_nodes_with_attributes(name=name)
        if len(nodes) == 0:
            raise Error('No node with name {}. ' + refer_to_faq_msg(51), name)
        elif len(nodes) > 1:
            raise Error('Multiple nodes with name {}'.format(name))
        else:
            return nodes[0]

    def get_op_nodes(self, **attrs):
        nodes = self.get_nodes_with_attributes(**dict(kind='op', **attrs))
        return [Node(self, node) for node in nodes]

    def get_data_nodes(self, has_value=None):
        """
        Returns list of data nodes.
        If has_value = True, returns data nodes with value
        If has_value = False, returns data nodes without value
        """
        data_nodes = [Node(self, node) for node in self.nodes() if Node(self, node).soft_get('kind') == 'data']
        return [node for node in data_nodes if has_value is None or node.has_valid('value') == has_value]

    def get_nodes_with_attributes(self, **attrs: dict):
        node_attrs = self.nodes(data=True)
        return [n for n, d in node_attrs if all(a in d.items() for a in attrs.items())]

    def unique_id(self, prefix: str = ""):
        """
        Generates a unique node id for a new node in a given graph.
        The optional string prefix can be specified.
        """
        # TODO thread safety?
        self.unique_id_count = max(self.unique_id_count, self.number_of_nodes()) + 1
        if prefix and not self.has_node(prefix):
            return str(prefix)
        while self.has_node(prefix + str(self.unique_id_count)):
            self.unique_id_count += 1
        return prefix + str(self.unique_id_count)

    def check_empty_graph(self, description: str):
        if len(self.nodes()) <= 1:
            raise Error(
                "Graph contains {} node after executing {}. It considered as error because resulting IR will be "
                "empty which is not usual".format(len(self.nodes()), description))

    def check_shapes_consistency(self):
        data_nodes = self.get_data_nodes()
        data_nodes_with_wrong_shapes = []
        for data_node in data_nodes:
            if not data_node.has('shape'):
                data_nodes_with_wrong_shapes.append((data_node.name, "no shape attribute"))
                continue
            if data_node.shape is not None and not isinstance(data_node.shape, np.ndarray):
                data_nodes_with_wrong_shapes.append((data_node.name, type(data_node.shape)))
        if len(data_nodes_with_wrong_shapes) > 0:
            raise Error("Graph contains data nodes ({}) with inconsistent shapes: {}".format(
                len(data_nodes_with_wrong_shapes),
                data_nodes_with_wrong_shapes
            ))

    def check_nodes_ports_are_consecutive(self):
        # Check that all operation nodes has consecutive ports indexes
        op_nodes = self.get_op_nodes()
        for node in op_nodes:
            for idx in range(len(node.in_ports())):
                if idx not in node.in_ports():
                    raise Error("Node {} has not consecutive in ports indexes: {}".format(node.name,
                                                                                          list(node.in_ports().keys())))
            for idx in range(len(node.out_ports())):
                if idx not in node.out_ports():
                    raise Error("Node {} has not consecutive out ports indexes: {}".format(node.name,
                                                                                           list(
                                                                                               node.out_ports().keys())))

    def dump_graph_for_graphviz(self, node_attrs: list = ['kind', 'op', 'shape', 'correct_data_layout', 'nchw_layout',
                                                          'internal_layer_id'],
                                edge_attrs: list = ['in', 'out'], nodes_to_dump: list = None,
                                save_to_svg=False, highlight_nodes: list = None):

        from openvino.tools.mo.ops.tensor_iterator import _get_internal_output_node_id, _get_internal_input_node_id

        fill_color = {'op': 'lightblue', 'data': 'whitesmoke', 'highlight': 'firebrick'}
        fill_color_by_type = {'Const': 'lightpink', 'Parameter': 'yellowgreen', 'TensorIterator': 'lemonchiffon'}
        style = {'op': 'filled,bold', 'data': 'filled,rounded'}

        subgraphs = {}
        if highlight_nodes is None:
            highlight_nodes = []

        def _subgraph_label(node_id, node_attrs: dict, attrs_to_print: list):
            subgraphs[node_id] = "cluster_{}".format(node_id)
            label = 'subgraph "cluster_{}" '.format(node_id) + '{\n'
            label += 'label = "{}"; \n'.format(node_id)
            label += 'color={}; \nstyle="filled,rounded";\n'.format(fill_color_by_type[node_attrs['op']])

            subgraph_name = node_attrs['sub_graphs']
            assert len(subgraph_name) == 1
            body = node_attrs[subgraph_name[0]].dump_graph_for_graphviz()
            body = body.split('\n')[2:-1]
            label += '\n'.join(body)
            label += '\n}\n'
            return label

        def _node_label(node_id, node_attrs: dict, attrs_to_print: list):
            label = str(node_id) + '\\n' + '\\n'.join([str(key) + '=' + str(node_attrs.get(key, 'None'))
                                                       for key in attrs_to_print if key in node_attrs])
            if node_attrs.get('type', '') == 'Const':
                if 'value' not in attrs_to_print and 'value' in node_attrs:
                    if node_attrs['value'] is not None:
                        label += '\\nvalue=\\"' + \
                                 ','.join([str(val) for val in node_attrs['value'].flatten()])[:40] + '\\"'
                    else:
                        label += '\\nvalue=None'
            return label

        def _dump_nodes_attrs():
            string = ''
            for node_id in nodes_to_dump:
                attrs = self.node[node_id]
                color = fill_color_by_type.get(attrs.get('type', ''), fill_color[attrs['kind']])

                if node_id in highlight_nodes or 'highlight' in node_attrs and node_attrs['highlight']:
                    color = fill_color['highlight']

                if attrs.get('op') == 'TensorIterator':
                    string += _subgraph_label(node_id, attrs, node_attrs)
                else:
                    string += '"{}" [fillcolor={} style="{}" shape=box label="{}"];\n'.format(
                        node_id, color, style[attrs['kind']], _node_label(node_id, attrs, node_attrs))
            return string

        def _dump_edges_attrs():
            string = ''
            for src_node_id, dst_node_id, attrs in self.edges(data=True):
                if src_node_id not in nodes_to_dump or dst_node_id not in nodes_to_dump:
                    continue

                if src_node_id in subgraphs:
                    edge_label = subgraphs[src_node_id]
                    edge_label_name = 'ltail'
                    src_node_id = _get_internal_output_node_id(self, src_node_id, attrs['external_port_id'])
                elif dst_node_id in subgraphs:
                    edge_label = subgraphs[dst_node_id]
                    edge_label_name = 'lhead'
                    dst_node_id = _get_internal_input_node_id(self, dst_node_id, attrs['external_port_id'])
                else:
                    edge_label = ' '.join(
                        [str(key) + '=' + str(attrs.get(key, 'None')) for key in edge_attrs if key in attrs])
                    edge_label_name = 'label'

                string += '"{}" -> "{}" [{} = "{}"];\n'.format(src_node_id, dst_node_id, edge_label_name, edge_label)
            return string

        log.debug("---- GRAPHVIZ OUTPUT STARTS ----")

        if nodes_to_dump is None:
            nodes_to_dump = self.nodes()

        string = '\ndigraph {\n'

        string += _dump_nodes_attrs()
        string += _dump_edges_attrs()

        string += '}'
        log.debug("---- GRAPHVIZ OUTPUT ENDS ----")

        if save_to_svg:
            try:
                import graphviz
                import os
                file_name = "{}_{}.txt".format(self.name.replace('/', '_'), 0)
                id = 1
                while os.path.exists(file_name):
                    file_name = "{}_{}.txt".format(self.name.replace('/', '_'), id)
                    id += 1
                with open(file_name, "w") as f:
                    f.write(string)
                graphviz.render('dot', 'svg', file_name)
                print('Graph was saved to {}.{}'.format(file_name, 'svg'))
            except ImportError:
                raise ImportError('Can\'t import graphviz')
            except Exception as e:
                raise Error('Can\'t save graph to svg') from e

        return string

    def print_graph_stat(self):
        log.debug('Number of nodes in graph: {}'.format(self.number_of_nodes()))
        log.debug('Number of edges in graph: {}'.format(len(list(self.edges()))))
        ops = collections.defaultdict(int)
        for _node in self.nodes():
            node = Node(self, _node)
            kind = node.kind if node.has('kind') else '<UNDEFINED>'
            if node.has('op'):
                ops['op/' + node.op] += 1
            else:
                ops[kind] += 1
            if node.has('shape') and np.any(node.shape == 0):
                log.error("Found bad shape: '{}' for node '{}'".format(node.shape, node.node))
        for k, v in ops.items():
            log.debug('   {} : {}'.format(k, v))

    def create_sub_graph_copy(self, nodes_to_extract: list):
        """
        Create new graph which is a sub-graph of the 'graph' that contains just nodes from 'nodes_to_extract' list. The
        returned sub-graph is a deep copy of the provided graph nodes.
        :param graph: graph to create a sub-graph from.
        :param nodes_to_extract: list of node names to extract.
        :return: new graph.
        """
        return self.subgraph(nodes_to_extract).copy()

    def create_edge(self, src_node: Node, dst_node: Node, out_port: int = 0, in_port: int = 0, edge_attrs: dict = None):
        """
        Creates edge from node 'src_node' from output with index 'out_port' to node 'dst_node' with input index 'in_port'.
        :param src_node: node to create edge from.
        :param dst_node: node to create edge to.
        :param out_port: the index of output tensor of the 'src_node'.
        :param in_port: the input index of the node 'dst_node'.
        :param edge_attrs: dictionary with edge attrs.
        :return: None
        """
        # edges must belong to the same graph
        assert src_node.graph is dst_node.graph
        graph = src_node.graph

        if edge_attrs is None:
            edge_attrs = dict()
        else:
            edge_attrs = edge_attrs.copy()
        edge_attrs.update(
            {'in': in_port, 'out': out_port, 'in_attrs': ['in', 'permutation'], 'out_attrs': ['out', 'permutation'],
             'data_attrs': ['fw_tensor_debug_info']})

        # TODO: in case if in_port do not exists, we should raise an Exception here
        graph.add_edges_from([(src_node.id, dst_node.id, edge_attrs)])

    def dfs(self, node_name: str, visited: set):
        """
        Implementation of the depth-first search algorithm starting from the specific node.
        :param graph: networkx graph to operate on.
        :param node_name: node name to start search from.
        :param visited: set of already visited nodes.
        :return: list of nodes in the DFS-visit order.
        """
        order = []
        stack = [node_name]
        while len(stack) != 0:
            node_name = stack[0]
            stack.pop(0)
            visited.add(node_name)
            has_child = False
            for _, out_node_name in self.out_edges(node_name):
                if out_node_name not in visited:
                    stack.insert(0, node_name)
                    stack.insert(0, out_node_name)
                    has_child = True
                    break
            if not has_child:
                order.append(node_name)
        return order

    def pseudo_topological_sort(self, reverse: bool = False):
        """
        The function performs topological sort but doesn't check for cycle existence. So it may produce wrong nodes order
        for some applications.
        :param graph: graph to pseudo-topologically sort.
        :param reverse: flag indicating whether need to reverse nodes order.
        :return: nodes in the topological sort if cycle doesn't exist and in pseudo-topological sort if not.
        """
        nodes_without_inputs = list()
        for node_name in self.nodes():
            if len(self.in_edges(node_name)) == 0:
                nodes_without_inputs.append(node_name)
        order = list()
        visited = set()
        for node_name in nodes_without_inputs:
            if node_name not in visited:
                order.extend(self.dfs(node_name, visited))

        order = [Node(self, node) for node in order]

        if reverse:
            return order
        else:
            return list(reversed(order))

    def pseudo_topological_sort_with_start_node(self, start_node: Node, reverse: bool = False):
        nodes_without_inputs = [start_node.soft_get('name')]
        visited = set()
        order = self.dfs(nodes_without_inputs[0], visited)

        order = [Node(self, node) for node in order]

        if reverse:
            return order
        else:
            return list(reversed(order))

    def clean_up(self, undead_node_types: list = None):
        if undead_node_types is None:
            undead_node_types = []

        if not getattr(self.graph['cmd_params'], 'static_shape', False):
            undead_node_types.extend(['ShapeOf', 'Shape', 'slice_like'])

        mark_output_reachable_nodes(self)
        shape_inference(self)
        mark_undead_nodes(self, undead_node_types)
        mark_const_producer_nodes(self)
        eliminate_dead_nodes(self)
        # Add Const op for constant data nodes
        add_constant_operations(self)

    def get_tensor_names_set(self):
        """
        Get set of tensor names of the graph.
        """
        tensor_names_set = set()
        for node in self.get_op_nodes():
            if self.stage is None:
                for out_edge_idx in node.out_edges():
                    out_edge = node.out_edge(out_edge_idx)
                    if "fw_tensor_debug_info" in out_edge:
                        for _, tensor_name in out_edge["fw_tensor_debug_info"]:
                            tensor_names_set.add(tensor_name)
            else:
                for _, port in node.out_ports().items():
                    tensor_names = port.get_tensor_names()
                    tensor_names_set = tensor_names_set.union(set(tensor_names))
        return tensor_names_set

    def has_tensor_name(self, tensor_name: str):
        """
        Checks if graph has tensor name.
        """
        # TODO: This can be optimized. Tensor names can be stored as set, which is initialized after model loading.
        names = self.get_tensor_names_set()
        return tensor_name in names


    def topological_sort(self, reverse: bool = False):
        sorted_node_ids = nx.topological_sort(self)

        sorted_nodes = [Node(self, node_id) for node_id in sorted_node_ids]

        if not reverse:
            return sorted_nodes
        else:
            return list(reversed(sorted_nodes))

    def set_node_attributes(self, name: str, values):
        return nx.set_node_attributes(self, values=values, name=name)


def fill_graph_with_nodes(graph, src_nodes, get_id: callable, get_attrs: callable):
    """
    Go over all nodes in src_nodes that should be enumerable and create new NX nodes
    using get_id and get_attrs functions to create node id and node attributes correspondingly.
    """
    for node in src_nodes:
        graph.add_node(get_id(node), **get_attrs(node))


def dict_includes_compare_attrs(attr, attr_probe):
    if callable(attr_probe) and not isinstance(attr_probe, type):
        return attr_probe(attr)
    else:
        res = (attr == attr_probe)
        # check if the result of comparison is a numpy scalar value which occur when attr is python scalar and
        # attr_probe is a numpy scalar
        if hasattr(res, 'ndim') and res.ndim == 0:
            return res.item()
        return res if isinstance(res, bool) else all(res)


def dict_includes(big: dict, sub_dict: dict, skip_attr_names=[]):
    """ Searches attributes from sub_dict in big and ensures that all values match.

        Entries in sub_dict can be of two types: callable or not callable. If callable is specified
        it is treated as probing function for attribute value from big dictionary by callable(attr) expression.
        If it is not callable, the values are compared with == operator.
    """
    return all(
        dict_includes_compare_attrs(big.get(attr, None), sub_dict[attr])
        for attr in sub_dict.keys() if attr not in skip_attr_names
    )


def add_opoutput(graph: Graph, node_name: str, port: int, cut: bool = True, keep_output_port: bool = False,
                 user_defined_name=None):
    """
    Creates and connects Result node to node_name port. Cuts existing port if requested.
    :param graph: graph to operate with
    :param node_name: name of existing node in the graph that we want to add Result to
    :param port: output port of node to connect Result to
    :param cut: determines way of operating with edge specified by node_name and port
    :param keep_output_port: special attribute determines if this operation is saved in IR or not
    :param user_defined_name: User defined operation name, which should be added to tensor names list
    """
    # we import it here because Op imports add_attrs_props and update_ie_fields from this file
    from openvino.tools.mo.ops.result import Result
    node = Node(graph, node_name)
    if cut and len(node.out_edges()) != 0:
        opoutput_node = Result(graph).create_node_on_port(node, port, {'name': node_name + '/sink_port_' + str(port),
                                                                       'keep_output_port': keep_output_port})
    else:
        opoutput_node = Result(graph).create_node([(node, port)], {'name': node_name + '/sink_port_' + str(port),
                                                                   'keep_output_port': keep_output_port})
        opoutput_node.in_edge()['data_attrs'] = ['fw_tensor_debug_info']

    if user_defined_name is not None and (graph.stage == 'front' or graph.stage is None):
        # Following code adds user_defined_name to tensor names list
        # Not applicable for middle stage
        prev_op_tensor_names = set()
        in_edge_attrs = opoutput_node.in_edge()
        if 'fw_tensor_debug_info' in in_edge_attrs:
            for _, tensor_name in opoutput_node.in_edge()['fw_tensor_debug_info']:
                prev_op_tensor_names.add(tensor_name)
        if user_defined_name not in prev_op_tensor_names:
            if graph.has_tensor_name(user_defined_name):
                log.warning('Could not add user defined output name {} to tensor names list of {} node as '
                            'graph contains tensor name with same name.'.format(user_defined_name,
                                                                                opoutput_node.soft_get('name')))
            else:
                if 'fw_tensor_debug_info' not in in_edge_attrs:
                    in_edge_attrs['fw_tensor_debug_info'] = []
                in_edge_attrs['fw_tensor_debug_info'].append([user_defined_name, user_defined_name])

    log.debug('Sink: {} for node {}'.format(opoutput_node.id, node_name))
    log.debug(str(graph.node[opoutput_node.id]))
    log.debug("Add edge from {} to {}".format(node_name, opoutput_node.id))
    return opoutput_node.id


# TODO implement merging for keys with dictionary values?
def merge_edge_props(attrs: dict, additional_attrs: dict):
    """
    Update edge attributes without changing 'in' and 'out' keys.
    It is necessary to copy edge attributes during merging of nodes when
    result of one subgraph call is passed as input to another subgraph call
    """
    result = attrs
    for (key, value) in additional_attrs.items():
        if key not in ['in', 'out']:
            if type(additional_attrs[key]) is list:
                if key not in result:
                    result[key] = []
                result[key].extend(additional_attrs[key])
                result[key] = list(set(result[key]))  # silly solution to find unique elements
            else:
                result[key] = value
    return result


def rename_node(node: Node, name):
    if not node.graph.get_nodes_with_attributes(name=name):
        node.name = name
    else:
        assert 'Node with name {} already exists'.format(name)


def rename_nodes(nodes: List[tuple]):
    for node, name in nodes:
        rename_node(node, name)


def get_edge_attribute_between_nodes(node1: Node, node2: Node, attr_name: str):
    """
    Gets edge attribute value between two nodes.
    This method is introduced for implementation of manual replacing of nodes attributes
    with tensor debug information. It is needed after removing of fake outputs.
    Also there are cases when graph transformations lead to mismatch of tensor name
    and input node, so manual attribute change is needed.
    This method should only be used during the front phase.
    And it is applicable only for cases when there is just one edge between two given nodes.
    """
    for edge_idx in node1.out_edges():
        edge = node1.out_edge(edge_idx)
        out_port = edge['out']
        out_node = node1.out_node(out_port)
        if out_node.id == node2.id:
            if attr_name in edge:
                return edge[attr_name]
    return None


def set_edge_attribute_between_nodes(node1: Node, node2: Node, attr_name: str, new_value):
    """
    Sets edge attribute value between two nodes.
    This method is introduced for implementation of manual replacing of nodes attributes
    with tensor debug information. It is needed after removing of fake outputs.
    Also there are cases when graph transformations lead to mismatch of tensor name
    and input node, so manual attribute change is needed.
    This method should only be used during the front phase.
    And it is applicable only for cases when there is just one edge between two given nodes.
    """
    for edge_idx in node1.out_edges():
        edge = node1.out_edge(edge_idx)
        out_port = edge['out']
        out_node = node1.out_node(out_port)
        if out_node.id == node2.id:
            edge[attr_name] = new_value

# All functions below are deprecated and will be removed in next release
# Please, use methods from Graph/Node classes instead


@deprecated_api(Graph)
def get_node_id_by_name(graph: Graph, name: str):
    return graph.get_node_id_by_name(name=name)


@deprecated_api(Graph)
def print_graph_stat(graph: Graph):
    return graph.print_graph_stat()


@deprecated_api(Graph)
def get_inputs_with_ports(graph: Graph, match, pattern_edges, input_names_in_pattern):
    """
    Front replacements of multi-input nodes should specify output port to add_node-like functions
    This function is a helper to get such information out of matched nodes
    :param graph: graph to operate on
    :param match: dictionary returned by matching function
    :param pattern_edges: edges that are specified in pattern
    :param input_names_in_pattern: names of matched nodes as they were specified in pattern that should be in
    resulting list
    :return: list of tuples of node and output port
    """
    return graph.get_inputs_with_ports(match=match,
                                       pattern_edges=pattern_edges,
                                       input_names_in_pattern=input_names_in_pattern)


@deprecated_api(Graph)
def dump_graph_for_graphviz(graph: Graph, node_attrs: list = ['kind', 'op', 'shape'],
                            edge_attrs: list = ['in', 'out'],
                            nodes_to_dump: list = None, save_to_svg=False):
    return graph.dump_graph_for_graphviz(node_attrs=node_attrs,
                                         edge_attrs=edge_attrs,
                                         nodes_to_dump=nodes_to_dump,
                                         save_to_svg=save_to_svg)


@deprecated_api(Graph)
def create_sub_graph_copy(graph: Graph, nodes_to_extract: list):
    """
    Create new graph which is a sub-graph of the 'graph' that contains just nodes from 'nodes_to_extract' list. The
    returned sub-graph is a deep copy of the provided graph nodes.
    :param graph: graph to create a sub-graph from.
    :param nodes_to_extract: list of node names to extract.
    :return: new graph.
    """
    return graph.create_sub_graph_copy(nodes_to_extract=nodes_to_extract)


@deprecated_api(Graph)
def get_graph_ops(graph: Graph):
    return graph.get_op_nodes()


@deprecated_api(Graph)
def check_empty_graph(graph: Graph, description: str):
    return graph.check_empty_graph(description=description)


@deprecated_api(Graph)
def create_edge(src_node: Node, dst_node: Node, out_port: int = 0, in_port: int = 0, edge_attrs: dict = None):
    """
    Creates edge from node 'src_node' from output with index 'out_port' to node 'dst_node' with input index 'in_port'.
    :param src_node: node to create edge from.
    :param dst_node: node to create edge to.
    :param out_port: the index of output tensor of the 'src_node'.
    :param in_port: the input index of the node 'dst_node'.
    :param edge_attrs: dictionary with edge attrs.
    :return: None
    """
    assert src_node.graph is dst_node.graph
    graph = src_node.graph
    return graph.create_edge(src_node=src_node, dst_node=dst_node, out_port=out_port, in_port=in_port,
                             edge_attrs=edge_attrs)


@deprecated_api(Graph)
def erase_node(node: Node):
    """
    Erases node from the graph and reconnect edges from input node(s) to output node(s)
    Produces assertion error if the node being removed has multiple inputs or outputs.
    The function can be used in the front phase only (when there are no data nodes in the graph).
    :param node: Node to erase
    """
    graph = node.graph
    return graph.erase_node(node)


@deprecated_api(Node)
def get_sorted_inputs(node: Node, control_flow: bool = False):
    return node.get_sorted_inputs(control_flow=control_flow)


@deprecated_api(Node)
def get_sorted_outputs(node: Node, control_flow: bool = False):
    return node.get_sorted_outputs(control_flow=control_flow)


@deprecated_api(Node)
def insert_node_after(node: Node, new_node: Node, node_out_port: int = 0):
    """
    Insert node 'new_node' after output with index 'node_out_port' of the node 'node'. All consumers of node 'node'
    output with index 'node_out_port' will be changed to consume node 'new_node'.
    The function should be used when graph doesn't contain data nodes yet.
    :param node: node after which new node should be inserted.
    :param new_node: node to be inserted.
    :param node_out_port: the output index for the node 'node' to insert
    :return: None
    """
    return node.insert_node_after(new_node=new_node, node_out_port=node_out_port)


@deprecated_api(Node)
def replace_node(old_node: Node, new_node: Node, new_node_out_port: int = None):
    """
    Replaces node 'old_node' with a node 'new_node' preserving edge attributes.
    :param old_node: node to be replaced.
    :param new_node: node to replace with.
    :return: None
    """
    return old_node.replace_node(new_node=new_node, new_node_out_port=new_node_out_port)


@deprecated_api(Node)
def copy_node(src_node: Node, new_attrs: dict = None, dst_graph: nx.MultiDiGraph = None):
    """ Copies node with all attributes (optionally updated) within the same graph or to different graph."""
    return src_node.copy_node(new_attrs=new_attrs, dst_graph=dst_graph)


@deprecated_api(Node)
def get_inputs(graph: Graph, node: str, edge_attr: dict = None, control_flow: bool = False):
    return Node(graph, node).get_inputs(edge_attr=edge_attr, control_flow=control_flow)


@deprecated_api(Node)
def get_outputs(graph: Graph, node: str, edge_attr: dict = None, control_flow: bool = False):
    return Node(graph, node).get_outputs(edge_attr=edge_attr, control_flow=control_flow)

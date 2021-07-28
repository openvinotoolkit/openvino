# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from copy import deepcopy

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.connection import Connection
from mo.utils.error import Error


class Port:
    class DataAccessor:
        def __init__(self):
            pass

    def __init__(self, node, idx: int, type: str, **kwargs):
        if type not in ['in', 'out']:
            raise Error("Inappropriate port type: {}".format(type))

        # We use self.__dict__ only to not to call __setattr__ method from __init__ function
        self.__dict__['node'] = node
        self.__dict__['idx'] = idx
        self.__dict__['type'] = type
        self.__dict__['data'] = self.DataAccessor()
        self.__dict__['control_flow'] = False
        self.__dict__.update(kwargs)

        self.data.get_shape = self._get_shape
        self.data.set_shape = self._set_shape

        self.data.get_value = self._get_value
        self.data.set_value = self._set_value

        self.data.get_attr = self._get_attr
        self.data.set_attr = self._set_attr

        self.data.has_valid = self._has_valid

    def __eq__(self, other):
        return (
                self.__class__ == other.__class__ and
                self.node.graph == other.node.graph and
                self.node.id == other.node.id and
                self.type == other.type and
                self.idx == other.idx
        )

    def __hash__(self):
        return hash((self.node.id, self.type, self.idx))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            result.__dict__[k] = v if k in ['graph', 'node'] else deepcopy(v)
        return result

    def __setattr__(self, key, value):
        edge = self.node.in_edge(self.idx, control_flow=self.control_flow) if self.type == 'in' else \
            self.node.out_edge(self.idx, control_flow=self.control_flow)
        edge[key] = value

    def __getattr__(self, item):
        edge = self.node.in_edge(self.idx, control_flow=self.control_flow) if self.type == 'in' else \
            self.node.out_edge(self.idx, control_flow=self.control_flow)
        if edge.get(item) is None:
            raise Error(
                "Edge from {}_port {} at node {} has no attribute {}".format(self.type, self.idx, self.node.name, item))
        return edge[item]

    def _create_data_if_necessary(self):
        if self.node.graph.stage == 'front':
            raise Error("_create_data_if_necessary method is not applicable for front Graph phase!")
        if self.type == 'in':
            raise Error("_create_data_if_necessary method is not applicable for 'in' Port type!")

        if self.idx not in self.node.out_nodes(control_flow=self.control_flow):
            from mo.ops.op import Op
            Op.create_data_node(self.node.graph, self.node, out_port=self.idx)
            self.node['need_shape_inference'] = True
        return self.node.out_node(self.idx, control_flow=self.control_flow)

    def _get_shape(self):
        if self.node.graph.stage == 'front':
            return None
        else:
            node_caller = self.node.in_node if self.type == 'in' else self.node.out_node
            return node_caller(self.idx, control_flow=self.control_flow).shape

    def _set_shape(self, shape):
        if self.node.graph.stage == 'front':
            raise NotImplementedError("set_shape not implemented for front phase")
        else:
            if self.type == 'in':
                assert self.node.in_node(self.idx, control_flow=self.control_flow).value is None
                self.node.in_node(self.idx, control_flow=self.control_flow).shape = int64_array(shape)
            else:
                data_node = self.node.out_node(self.idx, control_flow=self.control_flow)
                assert data_node.value is None or \
                       np.array_equal(data_node.soft_get('force_shape', data_node.shape), int64_array(shape))
                self.node.out_node(self.idx, control_flow=self.control_flow).shape = int64_array(shape)

    def _get_value(self):
        if self.node.graph.stage == 'front':
            return None
        else:
            if self.type == 'in':
                if self.idx in self.node.in_nodes(control_flow=self.control_flow) and \
                        self.node.in_node(self.idx, control_flow=self.control_flow).has_valid('value'):
                    return self.node.in_node(self.idx, control_flow=self.control_flow).value
            else:
                if self.idx in self.node.out_nodes(control_flow=self.control_flow) and \
                        self.node.out_node(self.idx, control_flow=self.control_flow).has_valid('value'):
                    return self.node.out_node(self.idx, control_flow=self.control_flow).value
        return None

    def _set_value(self, value):
        if self.node.graph.stage == 'front':
            raise Error("set_value is not applicable for graph front phase")
        else:
            data_node_caller = self.node.in_node if self.type == 'in' else self.node.out_node
            data_node = data_node_caller(self.idx, control_flow=self.control_flow)
            const_node = data_node.in_node(control_flow=self.control_flow) if self.type == 'in' else self.node

            force_shape = data_node.soft_get('force_shape', const_node.soft_get('force_shape', None))
            shape = int64_array(value.shape if force_shape is None else force_shape)

            # Set value to data node
            data_node.value = value
            data_node.shape = shape

            # Set value to constant producer
            if const_node.soft_get('type') == 'Const':
                const_node.value = value
                const_node.shape = shape

    def _get_attr(self, item: str):
        if self.node.graph.stage == 'front':
            return None
        else:
            if self.type == 'in':
                if self.idx in self.node.in_nodes(control_flow=self.control_flow) and \
                        self.node.in_node(self.idx, control_flow=self.control_flow).has_valid(item):
                    return self.node.in_node(self.idx, control_flow=self.control_flow)[item]
            else:
                if self.idx in self.node.out_nodes(control_flow=self.control_flow) and \
                        self.node.out_node(self.idx, control_flow=self.control_flow).has_valid(item):
                    return self.node.out_node(self.idx, control_flow=self.control_flow)[item]
        return None

    def _set_attr(self, item, value):
        raise NotImplementedError()

    def get_in_edge_attrs(self, data=False):
        assert self.type == 'in'
        for u, v, d in list(self.node.graph.in_edges(self.node.id, data=True)):
            if d['in'] == self.idx:
                edge_attrs = self.node.graph.get_edge_data(u, v)
                for key in edge_attrs:
                    if edge_attrs[key]['in'] == self.idx:
                        if data:
                            return edge_attrs[key], u, v, key
                        else:
                            return edge_attrs[key]
        if data:
            return None, None, None, None
        else:
            return None

    def _has_valid(self, item):
        if self.node.graph.stage == 'front':
            raise NotImplementedError
        else:
            if self.type == 'in':
                if self.idx in self.node.in_nodes(control_flow=self.control_flow) and \
                        self.node.in_node(self.idx, control_flow=self.control_flow).has_valid(item):
                    return True
            else:
                if self.idx in self.node.out_nodes(control_flow=self.control_flow) and \
                        self.node.out_node(self.idx, control_flow=self.control_flow).has_valid(item):
                    return True
        return False

    def disconnected(self):
        # This method returns False if port connected with some other port
        # otherwise it returns True

        if self.type == 'in':
            return self.get_source() is None
        else:
            return len(self.get_destinations()) == 0

    def get_source(self):
        # This method returns Port object that is producer (source) port for out port.
        # In case if out port has no source port return None

        assert self.type != 'out', "Can't get source for output port at {} node".format(self.node.name)

        from mo.graph.graph import Node
        producer_ports = []

        has_producer = False
        if self.node.graph.stage == 'front':
            for n, d in self.node.get_inputs(control_flow=self.control_flow):
                if d['in'] == self.idx:
                    node = Node(self.node.graph, n)
                    producer_ports.append(node.out_port(d['out'], control_flow=self.control_flow))
                    has_producer = True
            if not has_producer:
                return None
        else:
            if self.idx not in self.node.in_nodes(control_flow=self.control_flow):
                return None

            in_data = self.node.in_node(self.idx, control_flow=self.control_flow)
            for n, d in in_data.get_inputs(control_flow=self.control_flow):
                node = Node(self.node.graph, n)
                producer_ports.append(node.out_port(d['out'], control_flow=self.control_flow))

        if len(producer_ports) != 1:
            if self.node.graph.strict_mode:
                raise Error('Something bad has happened with graph! Data node "{}" has {} producers'.format(
                    self.node.id, len(producer_ports)))
            else:
                return None
        return producer_ports[0]

    def get_destination(self):
        # This method returns Port that is consumer (destination) port for in port.
        # In case if in port has no consumer return None

        consumer_ports = self.get_destinations()
        if not consumer_ports:
            return None

        if len(consumer_ports) > 1:
            raise Error("The number of destinations for {} node at {} port is {}".format(self.node.name,
                                                                                         self.idx,
                                                                                         len(consumer_ports)))
        return consumer_ports[0]

    def get_destinations(self):
        assert self.type != 'in', "Can't get destinations for input port at {} node".format(self.node.name)

        from mo.graph.graph import Node
        consumer_ports = []
        if self.node.graph.stage == 'front':
            producer_node = self.node
        else:
            # In case if node has no output data node in given port, we return None
            if self.idx not in self.node.out_nodes(control_flow=self.control_flow):
                return []
            producer_node = self.node.out_node(self.idx, control_flow=self.control_flow)

        for n, d in producer_node.get_outputs(edge_attr={'out': self.idx} if self.node.graph.stage == 'front' else None,
                                              control_flow=self.control_flow):
            node = Node(self.node.graph, n)
            consumer_ports.append(node.in_port(d['in'], control_flow=self.control_flow))
        return consumer_ports

    def get_tensor_names(self, port_renumber: bool = False):
        """
        Gets sorted tensor names list.
        :param port_renumber: defines whether data node index should be calculated considering port renumbering.
        """
        tensor_debug_info = self.get_tensor_debug_info(port_renumber)
        tensor_names_list = []
        for attr in tensor_debug_info:
            if attr is not None and len(attr) >= 2:
                tensor_name = attr[1]
                if tensor_name is not None and len(tensor_name) > 0:
                    tensor_names_list.append(tensor_name.replace(',', '\\,'))
        return sorted(tensor_names_list)

    def get_tensor_debug_info(self, port_renumber: bool = False):
        """
        Gets tensor debug info attribute.
        :param port_renumber: defines whether data node index should be calculated considering port renumbering.
        """
        def get_tensor_debug_info_from_attrs(attrs):
            if 'fw_tensor_debug_info' in attrs:
                if attrs['fw_tensor_debug_info'] is not None:
                    return attrs['fw_tensor_debug_info']
            return []

        assert self.type != 'in', "Can't get tensor debug info for input port at {} node".format(self.node.name)

        fw_debug_info = []
        if self.node.graph.stage == 'front':
            if self.idx in self.node.out_edges():
                out_edge = self.node.out_edge(self.idx)
                fw_debug_info += get_tensor_debug_info_from_attrs(out_edge)
        else:
            # before port renumbering we use sequential numbering
            node_idx = self.idx
            if port_renumber:
                if self.node.type != 'Const':
                    # after port renumbering port indices start from zero,
                    # but data node indices remain the same
                    node_idx = self.idx + len(self.node.in_nodes())

            if node_idx in self.node.out_nodes():
                out_node = self.node.out_node(node_idx)
                fw_debug_info += get_tensor_debug_info_from_attrs(out_node.attrs())
        return fw_debug_info


    def disconnect(self):
        if self.type == 'out':
            consumer_ports = self.get_destinations()
            if self.node.graph.stage == 'front':
                for port in consumer_ports:
                    self.node.graph.remove_edge(self.node.id, port.node.id)
            else:
                for port in consumer_ports:
                    src_node = port.node.in_node(port.idx).id
                    dst_node = port.node.id
                    for key, val in self.node.graph.get_edge_data(src_node, dst_node).items():
                        if val['in'] == port.idx:
                            self.node.graph.remove_edge(src_node, dst_node, key=key)
                            break
        else:
            source_port = self.get_source()
            if source_port is None:
                return
            for u, v, d in list(self.node.graph.in_edges(self.node.id, data=True)):
                if d['in'] == self.idx:
                    for key in self.node.graph.get_edge_data(u, v):
                        if self.node.graph.get_edge_data(u, v)[key]['in'] == self.idx:
                            self.node.graph.remove_edge(u, v, key=key)
                            return

    def get_connection(self):
        if self.type == 'in':
            return Connection(self.node.graph, self.get_source(), [self], control_flow=self.control_flow)
        else:
            return Connection(self.node.graph, self, self.get_destinations(), control_flow=self.control_flow)

    def connect(self, port):
        if self.type == 'in':
            self.get_connection().set_source(port)
        else:
            self.get_connection().add_destination(port)

    def _get_data_type(self):
        """
        Internal method which does not raise with error if the data type is not known.
        Check value of the data node to determine input port data type as well as the respective value in the
        '_out_port_data_type' dictionary.
        :return: The data type or None if it is not defined
        """
        node = self.node
        if self.type == 'out':
            if node.has_valid('_out_port_data_type') and self.idx in node._out_port_data_type:
                return node._out_port_data_type[self.idx]

            # check the data type of the output data node
            value = self.data.get_value()
            value_data_type = value.dtype if value is not None else None
            if value_data_type is not None:
                value_data_type = value.dtype if value is not None else None
                log.debug('The precision of the output port {} of node {} is determined from the data node as {}'
                          ''.format(self.idx, self.node.name, value_data_type))
                return value_data_type
            return None
        else:
            # check the data type of the input data node
            value = self.data.get_value()
            value_data_type = value.dtype if value is not None else None
            if value_data_type is not None:
                log.debug('The precision of the input port {} of node {} is determined from the data node as {}'
                          ''.format(self.idx, self.node.name, value_data_type))

            # The 'get_source' method raises an error if there is no producer op node for the input port. But here we
            # don't want to do this, so we temporary disable graph strict mode
            old_strict_mode_value = node.graph.strict_mode
            node.graph.strict_mode = False
            source_port = self.get_source()
            source_port_data_type = None
            if source_port is not None:
                source_port_data_type = source_port._get_data_type()
            node.graph.strict_mode = old_strict_mode_value

            # check for the data node and port data type inconsistency. TODO should we raise an error here?
            if value_data_type is not None and source_port_data_type is not None and \
                    value_data_type != source_port_data_type:
                log.warning('Inconsistent data type of the data node and port attribute for port {} of node {}: {} vs '
                            '{}. Return data type of the data node.'.format(self.idx, self.node.name,
                                                                            value_data_type, source_port_data_type))
            # the source port data type has higher priority over the value data type because the MO calculates values in
            # I64 precision for shapes but not all IE plugins support I64, so we should trust data type infer functions
            return source_port_data_type if source_port_data_type is not None else value_data_type

    def get_data_type(self):
        data_type = self._get_data_type()
        if data_type is None:
            raise Error('The data type for {} port {} of node {} is not defined'.format(self.type, self.idx,
                                                                                        self.node.name))
        return data_type

    def is_data_type_defined(self):
        """
        Check if the data-type is already defined for the port.
        :return: the result of the check
        """
        return self._get_data_type() is not None

    def set_data_type(self, data_type, override=False):
        assert self.type == 'out', 'The method can be called for output ports only'
        node = self.node
        if not node.has_valid('_out_port_data_type'):
            node['_out_port_data_type'] = {}
        if self.idx in node._out_port_data_type and data_type != node._out_port_data_type[self.idx] and not override:
            raise Error('Trying to override data type for output port {} of operation {}: from {} to {}'.format(
                self.idx, node.name, node._out_port_data_type[self.idx], data_type))
        node._out_port_data_type[self.idx] = data_type

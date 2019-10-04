"""
 Copyright (c) 2019 Intel Corporation

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
import numpy as np
from copy import deepcopy

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
            if self.type == 'in':
                return self.node.in_node(self.idx, control_flow=self.control_flow).shape
            else:
                return self.node.out_node(self.idx, control_flow=self.control_flow).shape

    def _set_shape(self, shape):
        if self.node.graph.stage == 'front':
            raise NotImplementedError("set_shape not implemented for front phase")
        else:
            if self.type == 'in':
                assert self.node.in_node(self.idx, control_flow=self.control_flow).value is None
                self.node.in_node(self.idx, control_flow=self.control_flow).shape = int64_array(shape)
            else:
                data_node = self.node.out_node(self.idx, control_flow=self.control_flow)
                assert data_node.value is None or np.array_equal(data_node.shape, int64_array(shape))
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
            if self.type == 'in':
                data_node = self.node.in_node(self.idx, control_flow=self.control_flow)
                const_node = data_node.in_node(control_flow=self.control_flow)
                # Set value to data node
                data_node.value = value
                data_node.shape = int64_array(value.shape)
                # Set value to constant producer
                const_node.value = value
                const_node.shape = int64_array(value.shape)
            else:
                self.node.out_node(self.idx, control_flow=self.control_flow).value = value
                self.node.out_node(self.idx, control_flow=self.control_flow).shape = int64_array(value.shape)
                if self.node.has_valid('type') and self.node.type == 'Const':
                    self.node.value = value
                    self.node.shape = int64_array(value.shape)

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

    def disconnect(self):
        if self.type == 'out':
            consumer_ports = self.get_destinations()
            if self.node.graph.stage == 'front':
                for port in consumer_ports:
                    self.node.graph.remove_edge(self.node.id, port.node.id)
            else:
                for port in consumer_ports:
                    self.node.graph.remove_edge(port.node.in_node(port.idx).id, port.node.id)
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

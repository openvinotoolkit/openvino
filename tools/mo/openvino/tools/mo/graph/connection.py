# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
from copy import deepcopy

from openvino.tools.mo.utils.error import Error


class Connection:
    def __init__(self, graph, source, destinations: list, control_flow=False):
        self.graph = graph
        self.source = source
        self.destinations = destinations
        self.control_flow = control_flow
        self.data = namedtuple('Data', ['get_value', 'get_shape'])
        self.data.get_value = self._get_value
        self.data.get_shape = self._get_shape

    def _get_value(self):
        if self.graph.stage == 'front':
            return None
        return self.source.node.out_node(self.source.idx, control_flow=self.control_flow).value

    def _get_shape(self):
        if self.graph.stage == 'front':
            return None
        return self.source.node.out_node(self.source.idx, control_flow=self.control_flow).shape

    @staticmethod
    def _get_new_tensor_debug_info(attributes_save_mode: str, source_attrs: dict, dest_attrs: dict):
        source_debug_info = []
        attr_name = 'fw_tensor_debug_info'
        if attr_name in source_attrs:
            source_debug_info = source_attrs[attr_name]
        dest_debug_info = []
        if attr_name in dest_attrs:
            dest_debug_info = dest_attrs[attr_name]
        if attributes_save_mode == "merge":
            return source_debug_info + dest_debug_info
        elif attributes_save_mode == "source":
            return source_debug_info
        else:
            return dest_debug_info

    @staticmethod
    def _update_tensor_debug_info(attrs: dict, new_value: list):
        if new_value is not None and len(new_value) > 0:
            attrs['fw_tensor_debug_info'] = new_value
        else:
            if 'fw_tensor_debug_info' in attrs:
                del attrs['fw_tensor_debug_info']

    def get_source(self):
        return self.source

    def get_destination(self):
        if self.destinations and len(self.destinations) > 1:
            raise Error("Connection has more than one destination: {}".format(len(self.destinations)))
        return self.destinations[0] if self.destinations else None

    def get_destinations(self):
        return self.destinations

    def set_source(self, port, attributes_save_mode=None):
        # In this method we are changing source for a connection with given port.
        # See detailed example below.
        #
        # SOURCE - Op1(out_port:0)
        #
        #                | Op4(in_port:0)
        # DESTINATIONS - | Op3(in_port:0)
        #                | Op2(in_port:0)
        #
        # NEW PORT - Op5(out_port:0)
        #
        #                               ,--->Op4(in_port:0)
        # CONNECTION                   ,--->Op3(in_port:0)
        #               Op1(out_port:0)--->Op2(in_port:0)
        #
        # When we set source for connection we disconnect existing source and reconnect all consumers to
        # the new given port with type='out'.
        #
        # UPDATED CONNECTION            ,--->Op4(in_port:0)
        #                              ,--->Op3(in_port:0)
        #               Op5(out_port:0)--->Op2(in_port:0)
        #
        # attributes_save_mode defines which attributes with tensor debug information should be
        # transferred to resulting connection.
        # 'source' - attributes are transferred from the outgoing edge (front phase) or
        # outgoing data node (middle/back phase) of the source of resulting connection.
        # 'dest' - attributes are transferred from the incoming edge (front phase) or
        # incoming data node (middle/back phase) of the destination of resulting connection.
        # 'merge' - attributes from source and destination are merged.

        if port.type == 'in':
            raise Error("Wrong port type in set_source method. Should be 'out' but given 'in'")

        if self.control_flow is True:
            raise Error("Cannot operate with connection with control_flow=True")

        if attributes_save_mode is None:
            attributes_save_mode = "merge"
            if self.source is not None:
                scr_node = self.source.node

                # Force "source" mode for "Parameter" source node, which preserves tensor names for
                # source node in connection.
                if scr_node.soft_get("type") == "Parameter":
                    attributes_save_mode = "source"

        if self.graph.stage == 'front':
            scr_node = port.node

            source_fw_names = []
            for dst_port in port.get_connection().destinations:
                edge_attrs, u, v, key = dst_port.get_in_edge_attrs(data=True)
                for attr in edge_attrs:
                    if attr == "fw_tensor_debug_info":
                        source_fw_names += edge_attrs[attr]
            # remove duplicates
            source_fw_names = list(set(source_fw_names))
            if not source_fw_names:
                attrs = {}
            else:
                attrs = {'fw_tensor_debug_info': source_fw_names}

            # Reconnecting all destinations as consumers to the source port preserving edge attrs
            for dst_port in self.destinations:
                edge_attrs, u, v, key = dst_port.get_in_edge_attrs(data=True)
                if u is not None:
                    edge_attrs['out'] = port.idx

                    new_tensor_info = self._get_new_tensor_debug_info(attributes_save_mode, attrs, edge_attrs)
                    self._update_tensor_debug_info(edge_attrs, new_tensor_info)

                    self.graph.remove_edge(u, v, key=key)
                    self.graph.add_edge(scr_node.id, v, **edge_attrs)
                else:
                    if attributes_save_mode == "dest":
                        attrs = {}
                    self.graph.create_edge(scr_node, dst_port.node, port.idx, dst_port.idx, edge_attrs=attrs)
        else:
            # Create out data node if not exists and mark node with need_shape_inference = True
            # In case if data node exists just use it.
            port._create_data_if_necessary()
            port_out_data = port.node.out_node(port.idx)

            attrs = {}
            if self.source is not None and self.source.idx in self.source.node.out_nodes():
                source_out_data = self.source.node.out_node(self.source.idx)
                # Copy attrs from source_out_data to port_out_data
                attrs = deepcopy(source_out_data.attrs())
                if attributes_save_mode != "source":
                    # Remove debug info
                    if 'fw_tensor_debug_info' in source_out_data.attrs():
                        del self.graph.node[source_out_data.id]['fw_tensor_debug_info']
                # Copy attrs to new data node
                for attr in attrs:
                    if attr != 'fw_tensor_debug_info':
                        port_out_data[attr] = attrs[attr]

            new_tensor_info = self._get_new_tensor_debug_info(attributes_save_mode, port_out_data.attrs(), attrs)
            self._update_tensor_debug_info(port_out_data.attrs(), new_tensor_info)

            for dst_port in self.destinations:
                edge_attrs, u, v, key = dst_port.get_in_edge_attrs(data=True)
                if u is not None:
                    self.graph.remove_edge(u, v, key=key)
                    self.graph.add_edge(port_out_data.id, v, **edge_attrs)
                else:
                    self.graph.add_edge(port_out_data.id, dst_port.node.id, **{'in': dst_port.idx})

    def set_destination(self, port, attributes_save_mode=None):
        # In this method we are changing destination for a connection with given port with type 'in'.
        # This method requires exactly one destination or empty destinations list.
        # See detailed example below.
        #
        # SOURCE - Op1(out_port:0)
        #
        # DESTINATIONS - Op2(in_port:0)
        #
        # NEW PORT - Op3(in_port:0)
        #
        # CONNECTION
        #               Op1(out_port:0)--->Op2(in_port:0)
        #
        # When we set destination for connection we disconnect destination port if exists and connect source to
        # the new given port with type='in'.
        #
        # UPDATED CONNECTION
        #
        #               Op1(out_port:0)--->Op3(in_port:0)
        #
        # attributes_save_mode defines which attributes with tensor debug information should be
        # transferred to resulting connection.
        # 'source' - attributes are transferred from the outgoing edge (front phase) or
        # outgoing data node (middle/back phase) of the source of resulting connection.
        # 'dest' - attributes are transferred from the incoming edge (front phase) or
        # incoming data node (middle/back phase) of the destination of resulting connection.
        # 'merge' - attributes from source and destination are merged.

        def check_and_remove_edge():
            if self.destinations:
                for destination in self.destinations:
                    edge_attrs, u, v, key = destination.get_in_edge_attrs(data=True)
                    if u is None:
                        raise Error(
                            "Broken Connection object! Destination (node:{}) is not connected to source.".format(
                                destination.node.name))
                    destination.disconnect()
                    return edge_attrs, key
            return {}, None

        if self.destinations and len(self.destinations) > 1:
            raise Error("set_destination applicable only for connections that has exactly one destination or "
                        "when there is no destinations")

        if port.type == 'out':
            raise Error("Wrong port type in set_destination method. Should be 'in' but given 'out'")

        if self.control_flow is True:
            raise Error("Cannot operate with connection with control_flow=True")

        if attributes_save_mode is None:
            attributes_save_mode = "merge"
            if self.source is not None:
                scr_node = self.source.node

                # Force "source" mode for "Parameter" source node, which preserves tensor names for
                # source node in connection.
                if scr_node.soft_get("type") == "Parameter":
                    attributes_save_mode = "source"

        if self.graph.stage == 'front':
            if self.source is not None:
                node = self.source.node
                source_attrs, _ = check_and_remove_edge()
                dest_attrs = port.get_in_edge_attrs() or {}

                edge_attrs = {}
                new_tensor_info = self._get_new_tensor_debug_info(attributes_save_mode, source_attrs, dest_attrs)
                self._update_tensor_debug_info(edge_attrs, new_tensor_info)

                self.graph.create_edge(node, port.node, out_port=self.source.idx, in_port=port.idx,
                                       edge_attrs=edge_attrs)
            self.destinations = [port]
        else:
            # create out node if not exists and mark node with need_shape_inference = True
            # in case if data node exists just use it as is
            if self.source is not None:
                data_node = self.source._create_data_if_necessary()
                edge_attrs, key = check_and_remove_edge()
                edge_attrs.update({'in': port.idx})

                dest_attrs = {}
                if port.idx in port.node.in_nodes():
                    dest_attrs = port.node.in_node(port.idx).attrs()

                new_tensor_info = self._get_new_tensor_debug_info(attributes_save_mode, data_node.attrs(), dest_attrs)
                self._update_tensor_debug_info(data_node.attrs(), new_tensor_info)

                self.graph.add_edge(data_node.id, port.node.id, key=key, **edge_attrs)
            self.destinations = [port]

    def add_destination(self, port):
        # In this method we are adding destination port with type 'in' for a connection.
        # See detailed example below.
        #
        # SOURCE - Op1(out_port:0)
        #
        # DESTINATIONS - Op2(in_port:0)
        #
        # NEW PORT - Op3(in_port:0)
        #
        # CONNECTION
        #               Op1(out_port:0)--->Op2(in_port:0)
        #
        # When we set destination for connection we disconnect destination port if exists and connect source to
        # the new given port with type='in'.
        #
        # UPDATED CONNECTION
        #                                 ,-->Op3(in_port:0)
        #               Op1(out_port:0)--->Op2(in_port:0)
        #

        if self.control_flow is True:
            raise Error("Cannot operate with connection with control_flow=True")

        if self.source is None:
            raise Error("Can not add destination for connection without source port!")

        if self.graph.stage == 'front':
            node = self.source.node
            self.graph.create_edge(node, port.node, out_port=self.source.idx, in_port=port.idx)
        else:
            data_node = self.source._create_data_if_necessary()
            self.graph.add_edge(data_node.id, port.node.id, **{'in': port.idx})

        self.destinations.append(port)

    def remove(self):
        # This method deletes all edges in connection. After that connection is not more accessible.
        # See detailed example below.
        #
        # SOURCE - Op1(out_port:0)
        #
        #                | Op4(in_port:0)
        # DESTINATIONS - | Op3(in_port:0)
        #                | Op2(in_port:0)
        #
        #                               ,--->Op4(in_port:0)
        # CONNECTION                   ,--->Op3(in_port:0)
        #               Op1(out_port:0)--->Op2(in_port:0)
        #
        # After removing edges connection will be empty
        #
        # REMOVED CONNECTION
        #            Op5(out_port:0)   Op4(in_port:0)  Op2(in_port:0)  Op3(in_port:0)
        #

        if self.destinations:
            for dst_port in self.destinations:
                dst_port.disconnect()
        self.source = None
        self.destinations = []

    def insert_node(self, new_node, attributes_save_mode: str = "merge"):
        assert len(new_node.out_ports()) == 1, 'The node {} has several output ports'.format(new_node.soft_get('name'))
        source_port = self.get_source()
        self.set_source(new_node.out_port(0), attributes_save_mode)
        source_port.connect(new_node.in_port(0))

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ConcatOdInputEraserAndPortsReconnect(MiddleReplacementPattern):
    """
    The transformation performs two actions with Concat operations:
    1. Disconnects empty inputs (input tensor has at least one input dimension equal to 0)
    2. Renumber Concat inputs to be 0, 1, 2,...
    """
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        for concat in graph.get_op_nodes(type='Concat'):
            for in_port in concat.in_ports().values():
                if not in_port.disconnected():
                    shape = in_port.data.get_shape()
                    assert shape is not None
                    if 0 in shape:
                        concat.delete_input_port(in_port.idx)

            connected_ports = [port for port_idx, port in sorted(concat.in_ports().items()) if not port.disconnected()]
            assert len(connected_ports), 'Concat "{}" have no inputs after removing inputs with 0 dimensions' \
                                         ''.format(concat.soft_get('name', concat.id))

            max_port_index = max([port_idx for port_idx in concat.in_ports().keys()])
            # re-connect input ports sequentially and remove all not used
            port_idx_to_connect = 0
            for port_idx in range(max_port_index + 1):
                if concat.is_in_port_connected(port_idx):
                    if port_idx != port_idx_to_connect:
                        concat.add_input_port(port_idx_to_connect, skip_if_exist=True)
                        concat.in_port(port_idx).get_connection().set_destination(concat.in_port(port_idx_to_connect))
                    port_idx_to_connect += 1
                elif port_idx in concat.in_ports():
                    concat.delete_input_port(port_idx)

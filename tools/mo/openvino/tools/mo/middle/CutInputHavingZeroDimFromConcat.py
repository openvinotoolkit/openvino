# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.concat import Concat


class CutInputHavingZeroDimFromConcat(MiddleReplacementPattern):
    """
    This transformation deletes inputs of Concat having zeros in their shapes, if not all inputs have such shapes.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('concat', dict(type='Concat'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        concat_node = match['concat']
        sources_of_ports = [concat_node.in_port(i).get_connection().get_source() for i in concat_node.in_ports()]
        # If 'concat' is ConcatV2 layer from TF, then this layer initially had input 'axis' as the last input.
        # But then this input was deleted and the attribute 'axis' was added. Hence, the last port source can
        # be None in such case.
        sources_of_ports = [s for s in sources_of_ports if s is not None]

        input_nodes = [s.node for s in sources_of_ports]
        if not all(n.has_valid('type') for n in input_nodes):
            return

        saved_ports = []
        disconnected_ports = []

        for port_num, node in enumerate(input_nodes):
            if node.soft_get('type') == 'Const' and len(node.shape) > 1 and any(i == 0 for i in node.shape):
                disconnected_ports.append(port_num)
            else:
                saved_ports.append(port_num)

        if not saved_ports or not disconnected_ports:
            return

        if len(saved_ports) == 1:
            before_concat = concat_node.in_port(saved_ports[0]).get_connection().get_source()
            concat_node.out_port(0).get_connection().set_source(before_concat)
            return

        new_concat_attrs = concat_node.attrs().copy()
        new_concat_attrs['name'] = concat_node.name + '/Concat_'
        new_concat_attrs['in_ports_count'] = len(saved_ports)
        new_concat_node = Concat(graph, attrs=new_concat_attrs).create_node()

        for new_port_num, old_port_num in enumerate(saved_ports):
            concat_node.in_port(old_port_num).get_connection().set_destination(new_concat_node.in_port(new_port_num))

        for p in disconnected_ports:
            concat_node.in_port(p).disconnect()

        concat_node.out_port(0).get_connection().set_source(new_concat_node.out_port(0))

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender
from openvino.tools.mo.utils.ir_reader.layer_to_class import copy_graph_with_ops


class LoopExtender(Extender):
    op = 'Loop'

    @staticmethod
    def extend(op: Node):
        def normalize_port_map(port_map: dict):
            for port in port_map:
                for elem in ['axis', 'stride', 'part_size', 'start', 'end', 'purpose']:
                    if port.get(elem) is None:
                        port[elem] = None

        assert op.has('body'), 'There is no "body" attribute in the Loop op {}.'.format(op.name)

        # Now op.body is an IREngine, we need to replace it with IREngine.graph
        op.body.graph.graph['cmd_params'] = op.graph.graph['cmd_params']
        op.body.graph.graph['ir_version'] = op.graph.graph['ir_version']
        op.body.graph.name = op.name + '/body'

        for node in op.body.graph.get_op_nodes():
            node['internal_layer_id'] = int(node.id)

        op.body = copy_graph_with_ops(op.body.graph)

        normalize_port_map(op.input_port_map)
        normalize_port_map(op.output_port_map)

        # the 'external_port_id' uses end-to-end numbering of ports, but at this moment it is separate for input and
        # output ports so we need to decrease the output por_id with a number of input ports
        for record in op.output_port_map:
            if record['external_port_id'] != -1:
                record['external_port_id'] -= len(op.in_ports())

        for edge in op.back_edges:
            edge['from_layer'] = edge['from-layer']
            edge['to_layer'] = edge['to-layer']

            edge['to_port'] = 0
            edge['from_port'] = 0

            del(edge['from-layer'])
            del(edge['to-layer'])

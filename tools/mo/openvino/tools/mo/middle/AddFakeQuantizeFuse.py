# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from typing import Dict

from openvino.tools.mo.middle.MulFakeQuantizeFuse import resolve_shared_inputs
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.fusing.helpers import get_tensor_in_port, get_value_in_port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class AddFakeQuantizeFuse(MiddleReplacementPattern):
    """ Fuses Add --> FakeQuantize sequence if possible
    """
    enabled = False

    def run_after(self):
        return []

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('preop', dict(op='Add', can_be_fused=True)),
                ('preoped', dict()),
                ('quantize', dict(op='FakeQuantize')),
            ],
            edges=[
                ('preop', 'preoped'),
                ('preoped', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        quantize = match['quantize']
        preop = match['preop']

        for i in [0, 1]:
            if preop.in_port(i).get_source().node.soft_get('type') in ['Convolution', 'Deconvolution', 'MatMul']:
                return

        tensor_port, value_port = get_tensor_in_port(preop), get_value_in_port(preop)

        if value_port is None or value_port.data.get_value() is None:
            log.debug('AddQuantizeFuse: cannot fuse because Add op has dynamic inputs')
            return

        # Direct modifications to quantize 1-st and 2-nd port inputs are performed.
        # So the data nodes at those inputs shouldn't have more than 1 consumer maximum 2 consumers to the same
        # quantize op (consumed by 1st and 2nd ports). So we duplicate FakeQuantize in_port 1, 2, 3, 4 data
        resolve_shared_inputs(node=quantize, port_ids_to_duplicate=[1, 2])

        quantize.in_port(1).data.set_value(quantize.in_port(1).data.get_value() - value_port.data.get_value())
        if quantize.in_node(1).id != quantize.in_node(2).id:
            quantize.in_port(2).data.set_value(quantize.in_port(2).data.get_value() - value_port.data.get_value())

        in_add_connection = quantize.in_port(0).get_source().node.in_port(0).get_connection()
        quantize.in_port(0).disconnect()
        in_add_connection.add_destination(quantize.in_port(0))

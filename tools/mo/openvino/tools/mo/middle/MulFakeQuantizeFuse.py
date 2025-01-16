# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from collections import defaultdict
from typing import Dict, List

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.passes.fusing.helpers import get_tensor_in_port, get_value_in_port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const


def resolve_shared_inputs(node: Node, port_ids_to_duplicate: List[int]):
    """
    Duplicates shared constants that are consumed by more than one node. 
    If constant is consumed by several ports of one node - no duplication gets done
    """
    graph = node.graph

    for port_id in port_ids_to_duplicate:
        dst_port_map = defaultdict(list)
        for dst in node.in_port(port_id).get_source().get_connection().get_destinations():
            dst_port_map[dst.node].append(dst.idx)
        del dst_port_map[node]
        value = node.in_port(port_id).data.get_value()
        if value is None:
            log.debug('Can not duplicate due no data for in_port {} of node {}'.format(port_id, node.name))
        for node, idxs in dst_port_map.items():
            const = Const(graph, {'value': mo_array(value),
                                  'name': node.soft_get('name', node.id) + '/duplicated_'}).create_node()
            for idx in idxs:
                node.in_port(idx).disconnect()
                const.out_port(0).connect(node.in_port(idx))
            const.infer(const)


class MulFakeQuantizeFuse(MiddleReplacementPattern):
    """ Fuses Mul --> FakeQuantize sequence if possible
    """
    enabled = False

    def run_after(self):
        return []

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('preop', dict(op='Mul', can_be_fused=True)),
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

        tensor_port, value_port = get_tensor_in_port(preop), get_value_in_port(preop)

        if value_port is None or value_port.data.get_value() is None:
            log.debug('MulQuantizeFuse: cannot fuse because Mul op has dynamic inputs')
            return

        mul_val = value_port.data.get_value()
        if np.any(mul_val <= 0):
            return

        # Direct modifications to quantize 1-st and 2-nd port inputs are performed.
        # So the data nodes at those inputs shouldn't have more than 1 consumer maximum 2 consumers to the same
        # quantize op (consumed by 1st and 2nd ports). So we duplicate FakeQuantize in_port 1, 2 data if needed
        resolve_shared_inputs(node=quantize, port_ids_to_duplicate=[1, 2])

        # TODO: need some special processing for values that exactly equal to threshold

        quantize.in_port(1).data.set_value(quantize.in_port(1).data.get_value() / mul_val)
        if quantize.in_node(1).id != quantize.in_node(2).id:
            quantize.in_port(2).data.set_value(quantize.in_port(2).data.get_value() / mul_val)

        # Reconnect Mul as it no longer needed for current FakeQuantize
        in_mul_connection = quantize.in_port(0).get_source().node.in_port(0).get_connection()
        quantize.in_port(0).disconnect()
        in_mul_connection.add_destination(quantize.in_port(0))

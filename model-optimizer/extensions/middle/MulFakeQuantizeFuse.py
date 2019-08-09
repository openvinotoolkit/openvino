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
import logging as log
from collections import defaultdict
from typing import Dict, List

import numpy as np

from mo.graph.graph import Graph, Node
from mo.middle.passes.conv import get_tensor_in_port, get_value_in_port
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


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
            const = Const(graph, {'value': np.array(value)}).create_node()
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
                ('preop', dict(op='Mul')),
                ('preoped', dict()),
                ('quantize', dict(op='FakeQuantize', keep_in_IR=True)),
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
        mul_val = value_port.data.get_value()
        if mul_val is None:
            log.debug('MulQuantizeFuse: cannot fuse because Mul op has dynamic inputs')
            return

        # Direct modifications to quantize 1-st and 2-nd port inputs are performed.
        # So the data nodes at those inputs shouldn't have more than 1 consumer maximum 2 consumers to the same
        # quantize op (consumed by 1st and 2nd ports). So we duplicate FakeQuantize in_port 1, 2 data if needed
        resolve_shared_inputs(node=quantize, port_ids_to_duplicate=[1, 2])

        # TODO: need some special processing for values that exactly equal to threshold

        # Need to flip output_low and output_high for those elements that have multiplier < 0
        if np.all(mul_val < 0):
            mi_o_node = quantize.in_port(3).get_source()
            ma_o_node = quantize.in_port(4).get_source()

            quantize.in_port(3).disconnect()
            quantize.in_port(4).disconnect()

            mi_o_node.connect(quantize.in_port(4))
            ma_o_node.connect(quantize.in_port(3))

        elif np.any(mul_val < 0):
            # Flipping values should be done on exclusive inputs of FakeQuantize node, so we duplicate them if needed
            resolve_shared_inputs(node=quantize, port_ids_to_duplicate=[3, 4])

            # Successful flipping will be done on broadcasted arrays
            mi_o_val = quantize.in_port(3).data.get_value()
            ma_o_val = quantize.in_port(4).data.get_value()
            mul_val, mi_o_val, ma_o_val = [np.array(a) for a in np.broadcast_arrays(mul_val, mi_o_val, ma_o_val)]

            neg_idx = np.where(mul_val < 0)
            mi_o_val[neg_idx], ma_o_val[neg_idx] = ma_o_val[neg_idx], mi_o_val[neg_idx]

            # TODO: revert broadcasting where unnecessary
            quantize.in_port(3).data.set_value(mi_o_val)
            quantize.in_port(4).data.set_value(ma_o_val)

        quantize.in_port(1).data.set_value(quantize.in_port(1).data.get_value() / mul_val)
        if quantize.in_node(1).id != quantize.in_node(2).id:
            quantize.in_port(2).data.set_value(quantize.in_port(2).data.get_value() / mul_val)

        # Reconnect Mul as it no longer needed for current FakeQuantize
        in_mul_connection = quantize.in_port(0).get_source().node.in_port(0).get_connection()
        quantize.in_port(0).disconnect()
        in_mul_connection.add_destination(quantize.in_port(0))

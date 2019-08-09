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
from typing import Dict

from extensions.middle.MulFakeQuantizeFuse import resolve_shared_inputs
from mo.graph.graph import Graph, Node
from mo.middle.passes.conv import get_tensor_in_port, get_value_in_port
from mo.middle.replacement import MiddleReplacementPattern


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
                ('preop', dict(op='Add')),
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
        add_val = value_port.data.get_value()
        if add_val is None:
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

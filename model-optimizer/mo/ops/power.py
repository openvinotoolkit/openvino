"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.graph.graph import Graph, Node
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.ops.op import Op


class AttributedPower(Op):
    op = 'AttributedPower'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': 'Power',

            'power': 1,
            'scale': 1,
            'shift': 0,

            'infer': self.infer,
            'type_infer': self.type_infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['power', 'scale', 'shift']

    @staticmethod
    def type_infer(node: Node):
        node.out_port(0).set_data_type(data_type_str_to_np(node.graph.graph['cmd_params'].data_type))

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)
        connected_inputs = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_inputs) == 1 and 0 in connected_inputs, \
            "AttributedPower should have 1 connected input port, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_inputs)

        assert node.has_valid('scale'), \
            'AttributedPower operation should have `scale` parameter set, but it doesn`t for node {}'.format(name)
        assert node.has_valid('shift'), \
            'AttributedPower operation should have `shift` parameter set, but it doesn`t for node {}'.format(name)
        assert node.has_valid('power'), \
            'AttributedPower operation should have `power` parameter set, but it doesn`t for node {}'.format(name)

        eltwise_infer(node, lambda a: np.power(a * node.scale + node.shift, node.power))

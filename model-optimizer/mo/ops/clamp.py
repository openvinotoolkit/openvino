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

from mo.graph.graph import Graph
from mo.ops.op import Op


class AttributedClamp(Op):
    op = 'AttributedClamp'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': 'Clamp',
            'op': self.op,
            'version': 'opset1',
            'infer': self.infer,
            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [
            'max',
            'min'
        ]

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]

        assert len(connected_in_ports) == 1 and connected_in_ports[0].idx == 0, \
            'AttributedClamp should have only one input, but it has {}'.format(len(connected_in_ports))
        assert node.has_valid('max') and node.has_valid('min'), \
            'Mandatory attributes `max` and `min` were not set for AttributedClamp node: `{}`'.format(name)
        assert node.max >= node.min, \
            'AttributedClamp max=={} is less than min=={} for node `{}`'.format(node.max, node.min, name)

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.clip(node.in_port(0).data.get_value(), node['min'], node['max']))
        else:
            node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())


class Clamp(Op):
    op = 'Clamp'

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': self.op,
            'infer': self.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)
        min_input_connected = node.has_port('in', 1) and not node.in_port(1).disconnected()
        max_input_connected = node.has_port('in', 2) and not node.in_port(2).disconnected()

        input_value = node.in_port(0).data.get_value()
        min_value = node.in_port(1).data.get_value() if min_input_connected else np.finfo(np.float32).min
        max_value = node.in_port(2).data.get_value() if max_input_connected else np.finfo(np.float32).max

        if input_value is not None and min_value is not None and max_value is not None:
            assert np.all(max_value >= min_value), \
                'Clamp max_value=={} is less than min_value=={} for node `{}`'.format(max_value, min_value, name)
            node.out_port(0).data.set_value(np.clip(node.in_port(0).data.get_value(), min_value, max_value))
        else:
            node.out_port(0).data.set_shape(node.in_port(0).data.get_shape())

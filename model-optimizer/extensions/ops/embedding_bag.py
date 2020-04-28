"""
 Copyright (c) 2020 Intel Corporation

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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op

import numpy as np


class EmbeddingBagOffsetsSum(Op):
    op = 'EmbeddingBagOffsetsSum'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,

            'in_ports_count': 5,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) >= 3 and 0 in connected_in_ports and 1 in connected_in_ports and \
               2 in connected_in_ports, "EmbeddingBag should have at least 3 connected input port, but it doesn't " \
                                        "for node: `{}`. Ports: {}".format(name, connected_in_ports)

        weights = node.in_port(0).data.get_value()
        assert weights is not None and len(weights.shape) >= 2
        input_shape = node.in_port(1).data.get_shape()
        assert input_shape is not None
        offsets_shape = node.in_port(2).data.get_shape()
        assert offsets_shape is not None and len(offsets_shape) == 1

        node.out_port(0).data.set_shape(np.concatenate((input_shape[0], weights.shape[1:]), dtype=np.int64))


class EmbeddingBagPackedSum(Op):
    op = 'EmbeddingBagPackedSum'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,

            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) >= 2 and 0 in connected_in_ports and 1 in connected_in_ports, \
            "EmbeddingBagPackedSum should have at least 2 connected input port, but it doesn't for node: `{}`. " \
            "Ports: {}".format(name, connected_in_ports)

        weights = node.in_port(0).data.get_value()
        assert weights is not None and len(weights.shape) >= 2
        input_shape = node.in_port(1).data.get_shape()
        assert input_shape is not None

        node.out_port(0).data.set_shape(np.concatenate((input_shape[0], weights.shape[1:]), dtype=np.int64))


class EmbeddingSegmentsSum(Op):
    op = 'EmbeddingSegmentsSum'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,

            'infer': self.infer,

            'in_ports_count': 6,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) >= 4 and 0 in connected_in_ports and 1 in connected_in_ports and \
               2 in connected_in_ports and 3 in connected_in_ports, \
            "EmbeddingSegmentsSum should have at least 4 connected input port, but it doesn't for node: `{}`. " \
            "Ports: {}".format(name, connected_in_ports)

        weights_shape = node.in_port(0).data.get_shape()
        assert len(weights_shape) >= 2
        num_segments = node.in_port(3).data.get_value()
        assert num_segments is not None, "EmbeddingSegmentsSum should have a constant num_segments provided, but it " \
                                         "doesn't for node: `{}`.".format(name)
        output_shape = int64_array(num_segments.tolist() + weights_shape[1:].tolist())
        node.out_port(0).data.set_shape(output_shape)

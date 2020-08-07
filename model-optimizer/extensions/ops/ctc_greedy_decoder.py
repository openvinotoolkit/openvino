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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class CTCGreedyDecoderOp(Op):
    op = 'CTCGreedyDecoder'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',

            'infer': self.infer,
            'reinterp_shape': True,

            'in_ports_count': 2,
            'out_ports_count': 1
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'ctc_merge_repeated'
        ]

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) == 2, \
            "Incorrect number of inputs for {} node".format(node_name)

        logits_shape = node.in_port(0).data.get_shape()
        sequence_mask_shape = node.in_port(1).data.get_shape()

        # check shapes of input tensors
        assert len(logits_shape) == 3 and len(sequence_mask_shape) == 2, \
            'Incorrect rank of some input tensor for {} node'.format(node_name)
        assert logits_shape[1] == sequence_mask_shape[1], \
            'Batch dimensions of input tensors must be the same for {} node'.format(node_name)
        assert logits_shape[0] == sequence_mask_shape[0], \
            'Time dimensions of input tensors must be the same for {} node'.format(node_name)

        batch_size = logits_shape[1]
        time_size = logits_shape[0]
        node.out_port(0).data.set_shape(int64_array([batch_size, time_size, 1, 1]))

"""
 Copyright (C) 2017-2021 Intel Corporation

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


class CTCGreedyDecoderSeqLenOp(Op):
    op = 'CTCGreedyDecoderSeqLen'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset6',

            'infer': self.infer,
            'reinterp_shape': True,

            'in_ports_count': 3,
            'out_ports_count': 2
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'merge_repeated',
            'classes_index_type',
            'sequence_length_type'
        ]

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        assert len(connected_in_ports) in [2, 3], \
            "Incorrect number of inputs for {} node".format(node_name)

        logits_shape = node.in_port(0).data.get_shape()
        sequence_len_shape = node.in_port(1).data.get_shape()
        if len(node.in_nodes()) == 3:
            blank_index_shape = node.in_port(2).data.get_shape()
            assert len(blank_index_shape) == 1, \
                'Incorrect rank of blank_index for {} node'.format(node_name)

        # check shapes of input tensors
        assert len(logits_shape) == 3, \
            'Incorrect rank of logits for {} node'.format(node_name)

        assert len(sequence_len_shape) == 1, \
            'Incorrect rank of sequence length tensor for {} node'.format(node_name)
        assert logits_shape[0] == sequence_len_shape[0], \
            'Batch dimensions of input tensors must be the same for {} node'.format(node_name)

        batch_size = logits_shape[0]
        time_size = logits_shape[1]
        node.out_port(0).data.set_shape(int64_array([batch_size, time_size]))
        #node.out_port(1).data.set_shape(int64_array([batch_size]))

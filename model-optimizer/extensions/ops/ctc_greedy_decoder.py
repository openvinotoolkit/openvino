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

from mo.graph.graph import Node, Graph
from mo.ops.op import Op
from mo.utils.error import Error
from mo.front.common.partial_infer.utils import int64_array

class CTCGreedyDecoderOp(Op):
    op = 'CTCGreedyDecoder'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset1',
            'in_ports_count': 2,
            'out_ports_count': 1,
            'infer': CTCGreedyDecoderOp.ctc_greedy_decoder_infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'ctc_merge_repeated'
        ]

    @staticmethod
    def ctc_greedy_decoder_infer(node: Node):
        output_shape = np.ones(4, dtype=np.int)
        assert node.in_port(0).data.get_shape()[1] == node.in_port(0).data.get_shape()[1], \
            'Batch for CTCGreedyDecoder should be the same in both inputs'
        output_shape[0] = node.in_port(0).data.get_shape()[1]
        output_shape[1] = node.in_port(0).data.get_shape()[0]

        node.out_port(0).data.set_shape(output_shape)

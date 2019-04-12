"""
 Copyright (c) 2018-2019 Intel Corporation

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


class CTCGreedyDecoderOp(Op):
    op = 'CTCGreedyDecoder'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
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
        outn = node.out_node(0)
        inn = node.in_node(0)
        inn2 = node.in_node(1)
        outn.shape = np.ones(4, dtype=np.int)
        assert inn.shape[1] == inn2.shape[1], 'Batch for CTCGreedyDecoder should be the same in both inputs'
        outn.shape[0] = inn.shape[1]
        outn.shape[1] = inn.shape[0]

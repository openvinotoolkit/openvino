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

from extensions.front.tf.CTCGreedyDecoder import CTCGreedyDecoderReplacement
from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class CTCGreedyDecoder_extender(Extender):
    op = 'CTCGreedyDecoder'

    @staticmethod
    def extend(op: Node):
        if op.graph.graph['cmd_params'].framework == 'tf':
            # We need to implement same infer as in TF front transformation
            op['infer'] = CTCGreedyDecoderReplacement.tf_greedy_decoder_infer
            op['old_infer'] = CTCGreedyDecoderOp.ctc_greedy_decoder_infer

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
from mo.front.extractor import FrontExtractorOp
from mo.ops.eltwise_ninputs_in_1 import EltwiseNin1
from mo.front.kaldi.utils import read_token_value


class ElementwiseProductComponentFrontExtractor(FrontExtractorOp):
    op = 'elementwiseproductcomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters

        indim = read_token_value(pb, b'<InputDim>')
        outdim = read_token_value(pb, b'<OutputDim>')
        num_inputs = indim / outdim

        attrs = {'num_inputs': int(num_inputs),
                 'operation': 'mul'}

        EltwiseNin1.update_node_stat(node, attrs)
        return __class__.enabled

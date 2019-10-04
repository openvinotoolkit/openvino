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
from mo.front.kaldi.loader.utils import collect_until_token, read_binary_integer32_token, read_binary_float_token
from extensions.ops.pnorm import PNormOp
from mo.utils.error import Error


class PNormComponentFrontExtractor(FrontExtractorOp):
    op = 'pnormcomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        try:
            collect_until_token(pb, b'<InputDim>')
        except Error:
            raise Error("<InputDim> was not found")
        in_dim = read_binary_integer32_token(pb)

        try:
            collect_until_token(pb, b'<OutputDim>')
        except Error:
            raise Error("<OutputDim> was not found")
        out_dim = read_binary_integer32_token(pb)

        assert in_dim % out_dim == 0

        group = in_dim / out_dim

        try:
            collect_until_token(pb, b'<P>')
        except Error:
            raise Error("<P> was not found")
        p = read_binary_float_token(pb)

        attrs = {
                 'group': group,
                 'p': p,
        }

        PNormOp.update_node_stat(node, attrs)
        return __class__.enabled

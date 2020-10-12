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

from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import read_binary_integer32_token, collect_until_token, \
    read_binary_float_token
from extensions.ops.identity import Identity


class GeneralDropoutComponentFrontExtractor(FrontExtractorOp):
    op = 'generaldropoutcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<BlockDim>')
        block_dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<TimePeriod>')
        time_period = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<DropoutProportion>')
        dropout_proporion = read_binary_float_token(pb)

        # collect_until_token(pb, b'<Continuous>')
        Identity.update_node_stat(node, {})

        return cls.enabled

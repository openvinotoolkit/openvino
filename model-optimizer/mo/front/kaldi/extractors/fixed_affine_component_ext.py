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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, collect_until_token
from mo.front.kaldi.utils import read_binary_matrix, read_binary_vector
from mo.ops.inner_product import InnerProduct
from mo.utils.error import Error


class FixedAffineComponentFrontExtractor(FrontExtractorOp):
    op = 'fixedaffinecomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        collect_until_token(pb, b'<LinearParams>')
        weights, weights_shape = read_binary_matrix(pb)
        tag = find_next_tag(pb)
        read_placeholder(pb, 1)
        if tag != '<BiasParams>':
            raise Error('FixedAffineComponent must contain BiasParams')
        biases = read_binary_vector(pb)

        mapping_rule = {
            'out-size': weights_shape[0],
            'layout': 'NCHW'
        }
        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        InnerProduct.update_node_stat(node, mapping_rule)
        return __class__.enabled

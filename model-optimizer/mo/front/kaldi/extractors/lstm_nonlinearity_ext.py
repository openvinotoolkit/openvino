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
from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import collect_until_token
from mo.front.kaldi.utils import read_binary_matrix
from mo.ops.lstmnonlinearity import LstmNonLinearity


class LSTMNonlinearityFrontExtractor(FrontExtractorOp):
    op = 'lstmnonlinearitycomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        collect_until_token(pb, b'<Params>')
        ifo_x_weights, ifo_x_weights_shape = read_binary_matrix(pb)

        mapping_rule = {}

        embed_input(mapping_rule, 1, 'i_weights', ifo_x_weights[0:1024])
        embed_input(mapping_rule, 2, 'f_weights', ifo_x_weights[1024:2048])
        embed_input(mapping_rule, 3, 'o_weights', ifo_x_weights[2048:])

        LstmNonLinearity.update_node_stat(node, mapping_rule)
        return __class__.enabled

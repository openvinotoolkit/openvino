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

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.utils import read_binary_vector, read_learning_info
from mo.ops.scale_shift import ScaleShiftOp


class AddShiftFrontExtractor(FrontExtractorOp):
    op = 'addshift'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        read_learning_info(pb)
        biases = read_binary_vector(pb)
        bias_term = True
        mapping_rule = {'bias_term': bias_term}
        embed_input(mapping_rule, 1, 'weights', np.ones(biases.shape))
        embed_input(mapping_rule, 2, 'biases', biases)
        ScaleShiftOp.update_node_stat(node, mapping_rule)
        return __class__.enabled

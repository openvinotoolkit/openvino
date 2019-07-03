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
from mo.front.kaldi.loader.utils import read_binary_integer32_token, collect_until_token
from mo.ops.scale_shift import ScaleShiftOp


class NormalizeComponentFrontExtractor(FrontExtractorOp):
    op = 'normalizecomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)
        target_rms = 1
        d_scaled = dim * target_rms ** 2
        in_norm = np.zeros([dim], np.float64)
        in_norm += 1.0 / d_scaled
        in_norm = np.maximum(in_norm, 2. ** (-66))  # pylint: disable=assignment-from-no-return
        in_norm = np.power(in_norm, -0.5)  # pylint: disable=assignment-from-no-return
        attrs = {}
        embed_input(attrs, 1, 'weights', in_norm)
        ScaleShiftOp.update_node_stat(node, attrs)
        return __class__.enabled

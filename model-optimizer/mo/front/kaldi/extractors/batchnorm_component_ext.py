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

import numpy as np

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import read_binary_bool_token, read_binary_integer32_token, collect_until_token, read_binary_float_token
from mo.front.kaldi.utils import read_binary_vector
from mo.ops.scale_shift import ScaleShiftOp
from mo.utils.error import Error


class BatchNormComponentFrontExtractor(FrontExtractorOp):
    op = 'batchnormcomponent'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters

        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<BlockDim>')
        block_dim = read_binary_integer32_token(pb)

        if block_dim != dim:
            raise Error("Dim is not equal BlockDim for BatchNorm is not supported")

        collect_until_token(pb, b'<Epsilon>')
        eps = read_binary_float_token(pb)

        collect_until_token(pb, b'<TargetRms>')
        target_rms = read_binary_float_token(pb)

        collect_until_token(pb, b'<TestMode>')
        test_mode = read_binary_bool_token(pb)

        if test_mode is not False:
            raise Error("Test mode True for BatchNorm is not supported")

        collect_until_token(pb, b'<StatsMean>')
        mean = read_binary_vector(pb)

        collect_until_token(pb, b'<StatsVar>')
        var = read_binary_vector(pb)

        scale = target_rms / np.sqrt(var + eps)

        shift = - target_rms * mean / np.sqrt(var + eps)
        attrs = {}
        embed_input(attrs, 1, 'weights', scale)
        embed_input(attrs, 2, 'biases', shift)
        ScaleShiftOp.update_node_stat(node, attrs)
        return __class__.enabled

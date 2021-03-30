"""
 Copyright (C) 2021 Intel Corporation

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
from mo.front.kaldi.loader.utils import collect_until_token
from mo.front.kaldi.utils import read_binary_matrix, read_binary_vector, read_binary_vector_of_pairs
from mo.front.kaldi.loader.utils import read_token_value
from mo.ops.timeheightconvolution import TimeHeightConvolutionComponent


class TimeHeightConvolutionFrontExtractor(FrontExtractorOp):
    op = 'timeheightconvolutioncomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        collect_until_token(pb, b'<ConvolutionModel>')
        in_shape = read_token_value(pb, b'<NumFiltersIn>')
        out_shape = read_token_value(pb, b'<NumFiltersOut>')
        height_in = read_token_value(pb, b'<HeightIn>')
        height_out = read_token_value(pb, b'<HeightOut>')
        height_subsample = read_token_value(pb, b'<HeightSubsampleOut>')
        collect_until_token(pb, b'<Offsets>')
        offsets = read_binary_vector_of_pairs(pb, read_token=False, dtype=np.int32)
        collect_until_token(pb, b'<RequiredTimeOffsets>')
        time_offsets = read_binary_vector(pb, read_token=False, dtype=np.int32)
        collect_until_token(pb, b'<LinearParams>')
        weights, w_shape = read_binary_matrix(pb)
        collect_until_token(pb, b'<BiasParams>')
        biases = read_binary_vector(pb)

        offsets = offsets.reshape([len(offsets)//2, 2])
        mapping_rule = {'height_subsample': height_subsample,
                        'height_in': height_in,
                        'height_out': height_out,
                        'offsets': offsets,
                        'time_offsets': time_offsets,
                        'out-size': out_shape}

        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        TimeHeightConvolutionComponent.update_node_stat(node, mapping_rule)
        return cls.enabled

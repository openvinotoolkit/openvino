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

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import read_binary_bool_token, read_binary_integer32_token, collect_until_token, \
    read_binary_float_token
from mo.front.kaldi.utils import read_binary_vector, read_binary_matrix
from mo.ops.tdnncomponent import TdnnComponent


class TdnnComponentFrontExtractor(FrontExtractorOp):
    op = 'tdnncomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<MaxChange>')
        max_change = read_binary_float_token(pb)

        collect_until_token(pb, b'<L2Regularize>')
        collect_until_token(pb, b'<LearningRate>')

        collect_until_token(pb, b'<TimeOffsets>')
        time_offsets = read_binary_vector(pb, False, np.int32)

        collect_until_token(pb, b'<LinearParams>')
        weights, weights_shape = read_binary_matrix(pb)
        collect_until_token(pb, b'<BiasParams>')
        bias_params = read_binary_vector(pb)

        collect_until_token(pb, b'<OrthonormalConstraint>')
        orthonormal_constraint = read_binary_float_token(pb)  # used only on training

        collect_until_token(pb, b'<UseNaturalGradient>')
        use_natural_grad = read_binary_bool_token(pb)  # used only on training
        collect_until_token(pb, b'<NumSamplesHistory>')
        num_samples_hist = read_binary_float_token(pb)

        collect_until_token(pb, b'<AlphaInOut>')
        alpha_in_out = read_binary_float_token(pb), read_binary_float_token(pb)  # for training, usually (4, 4)

        # according to Kaldi documentation http://kaldi-asr.org/doc/classkaldi_1_1nnet3_1_1TdnnComponent.html#details
        # it looks like it's used only during training (but not 100% sure)
        collect_until_token(pb, b'<RankInOut>')
        rank_in_out = read_binary_integer32_token(pb), read_binary_integer32_token(pb)

        biases = np.array(bias_params) if len(bias_params) != 0 else None
        attrs = {
            'weights': np.reshape(weights, weights_shape),
            'biases': biases,
            'time_offsets': time_offsets,
        }
        TdnnComponent.update_node_stat(node, attrs)
        return cls.enabled

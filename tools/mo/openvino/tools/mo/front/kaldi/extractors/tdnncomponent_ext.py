# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_binary_bool_token, read_binary_integer32_token, collect_until_token, \
    read_binary_float_token
from openvino.tools.mo.front.kaldi.utils import read_binary_vector, read_binary_matrix
from openvino.tools.mo.ops.tdnncomponent import TdnnComponent


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

        biases = mo_array(bias_params) if len(bias_params) != 0 else None
        attrs = {
            'weights': np.reshape(weights, weights_shape),
            'biases': biases,
            'time_offsets': time_offsets,
        }
        TdnnComponent.update_node_stat(node, attrs)
        return cls.enabled

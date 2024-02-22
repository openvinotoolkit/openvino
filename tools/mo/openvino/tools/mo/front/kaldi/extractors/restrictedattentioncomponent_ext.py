# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_binary_bool_token, \
    read_binary_integer32_token, collect_until_token, read_binary_float_token
from openvino.tools.mo.front.kaldi.utils import read_binary_vector, read_binary_matrix
from openvino.tools.mo.ops.restrictedattentioncomponent import RestrictedAttentionComponent


class RestrictedAttentionComponentFrontExtractor(FrontExtractorOp):
    """
    This class is used for extracting attributes of RestrictedAttention Kaldi operator.
    """
    op = 'restrictedattentioncomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        """
        This method extracts attributes of RestrictedAttention operator from Kaldi model.
        Description of all attributes can be found in the operator documentation:
        https://kaldi-asr.org/doc/classkaldi_1_1nnet3_1_1RestrictedAttentionComponent.html
        """
        params = node.parameters

        attrs = {}

        collect_until_token(params, b'<NumHeads>')
        attrs['num_heads'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<KeyDim>')
        attrs['key_dim'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<ValueDim>')
        attrs['value_dim'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<NumLeftInputs>')
        attrs['num_left_inputs'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<NumRightInputs>')
        attrs['num_right_inputs'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<TimeStride>')
        attrs['time_stride'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<NumLeftInputsRequired>')
        attrs['num_left_inputs_required'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<NumRightInputsRequired>')
        attrs['num_right_inputs_required'] = read_binary_integer32_token(params)

        collect_until_token(params, b'<OutputContext>')
        attrs['output_context'] = read_binary_bool_token(params)

        collect_until_token(params, b'<KeyScale>')
        attrs['key_scale'] = read_binary_float_token(params)

        collect_until_token(params, b'<StatsCount>')
        attrs['stats_count'] = read_binary_float_token(params)

        collect_until_token(params, b'<EntropyStats>')
        entropy_stats = read_binary_vector(params)
        attrs['entropy_stats'] = mo_array(
            entropy_stats) if len(entropy_stats) != 0 else None

        collect_until_token(params, b'<PosteriorStats>')
        posterior_stats, posterior_stats_shape = read_binary_matrix(params)
        attrs['posterior_stats'] = np.reshape(
            posterior_stats, posterior_stats_shape)

        RestrictedAttentionComponent.update_node_stat(node, attrs)
        return cls.enabled

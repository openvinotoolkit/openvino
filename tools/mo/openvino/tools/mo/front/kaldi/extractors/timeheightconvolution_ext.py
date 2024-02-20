# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token, read_token_value
from openvino.tools.mo.front.kaldi.utils import read_binary_matrix, read_binary_vector, read_binary_vector_of_pairs
from openvino.tools.mo.ops.timeheightconvolution import TimeHeightConvolutionComponent


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
        weights, _ = read_binary_matrix(pb)
        collect_until_token(pb, b'<BiasParams>')
        biases = read_binary_vector(pb)

        offsets = offsets.reshape([len(offsets)//2, 2])
        mapping_rule = {  # stride for h axis
                        'height_subsample': height_subsample,
                        # input dimension for h axis
                        'height_in': height_in,
                        # output dimension for h axis
                        'height_out': height_out,
                        # input dimension for channel axis
                        'in_channels': in_shape,
                        # output dimension for channel axis
                        'out_channels': out_shape,
                        # array with pairs like the following
                        # [ (-1, -1) (-1, 0) (-1, 1)
                        #   (0, -1)  (0, 0)  (0, 1)
                        #   (1, -1)  (1, 0)  (1, 1)]
                        #  it means that kernel 3x3 will be applied to calculate current value of output
                        'offsets': offsets,
                        # required time offsets to calculate current convolution
                        # time_offsets = [-1, 0, 1] for previous example means no padding for time axis and
                        # 3 values should be prepared
                        # time_offsets = [0] means zero padding [1, 1] for time axis
                        'time_offsets': time_offsets,
                        'out-size': out_shape * height_out}

        embed_input(mapping_rule, 1, 'weights', weights)
        embed_input(mapping_rule, 2, 'biases', biases)

        TimeHeightConvolutionComponent.update_node_stat(node, mapping_rule)
        return cls.enabled

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.caffe.extractors.utils import embed_input
from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.loader.utils import collect_until_token, read_binary_float_token, read_binary_integer32_token
from mo.front.kaldi.utils import read_binary_vector
from mo.ops.scale_shift import ScaleShiftOp


class BatchNormComponentFrontExtractor(FrontExtractorOp):
    op = 'batchnormcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<BlockDim>')
        block_dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<Epsilon>')
        eps = read_binary_float_token(pb)

        collect_until_token(pb, b'<TargetRms>')
        target_rms = read_binary_float_token(pb)

        collect_until_token(pb, b'<StatsMean>')
        mean = read_binary_vector(pb)

        collect_until_token(pb, b'<StatsVar>')
        var = read_binary_vector(pb)

        scale = target_rms / np.sqrt(var + eps)

        shift = - target_rms * mean / np.sqrt(var + eps)

        scale = np.tile(scale, dim // block_dim)
        shift = np.tile(shift, dim // block_dim)

        attrs = {'out-size': dim}
        embed_input(attrs, 1, 'weights', scale)
        embed_input(attrs, 2, 'biases', shift)

        ScaleShiftOp.update_node_stat(node, attrs)
        return cls.enabled

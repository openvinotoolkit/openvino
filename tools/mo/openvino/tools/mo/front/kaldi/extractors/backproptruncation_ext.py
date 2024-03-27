# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_binary_float_token, read_binary_integer32_token, collect_until_token
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp


class BackPropTrancationFrontExtractor(FrontExtractorOp):
    op = 'backproptruncationcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<Scale>')
        scale = read_binary_float_token(pb)

        # TODO add real batch here
        attrs = {}
        embed_input(attrs, 1, 'weights', np.full([dim], scale))
        ScaleShiftOp.update_node_stat(node, attrs)
        return cls.enabled

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.pnorm import PNormOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token, read_binary_integer32_token, read_binary_float_token
from openvino.tools.mo.utils.error import Error


class PNormComponentFrontExtractor(FrontExtractorOp):
    op = 'pnormcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        try:
            collect_until_token(pb, b'<InputDim>')
        except Error:
            raise Error("<InputDim> was not found")
        in_dim = read_binary_integer32_token(pb)

        try:
            collect_until_token(pb, b'<OutputDim>')
        except Error:
            raise Error("<OutputDim> was not found")
        out_dim = read_binary_integer32_token(pb)

        assert in_dim % out_dim == 0

        group = in_dim / out_dim

        try:
            collect_until_token(pb, b'<P>')
        except Error:
            raise Error("<P> was not found")
        p = read_binary_float_token(pb)

        attrs = {
                 'group': group,
                 'p': p,
        }

        PNormOp.update_node_stat(node, attrs)
        return cls.enabled

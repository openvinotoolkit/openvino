# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import read_binary_integer32_token, collect_until_token, \
    read_binary_float_token
from openvino.tools.mo.ops.identity import Identity


class GeneralDropoutComponentFrontExtractor(FrontExtractorOp):
    op = 'generaldropoutcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        collect_until_token(pb, b'<Dim>')
        dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<BlockDim>')
        block_dim = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<TimePeriod>')
        time_period = read_binary_integer32_token(pb)

        collect_until_token(pb, b'<DropoutProportion>')
        dropout_proporion = read_binary_float_token(pb)

        # collect_until_token(pb, b'<Continuous>')
        Identity.update_node_stat(node, {})

        return cls.enabled

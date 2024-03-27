# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.normalize import NormalizeOp
from openvino.tools.mo.front.caffe.extractors.utils import embed_input
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.loader.utils import collect_until_token, read_binary_bool_token, read_binary_integer32_token, \
    read_binary_float_token
from openvino.tools.mo.utils.error import Error


class NormalizeComponentFrontExtractor(FrontExtractorOp):
    op = 'normalizecomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        try:
            collect_until_token(pb, b'<Dim>')
        except Error:
            try:
                pb.seek(0)
                collect_until_token(pb, b'<InputDim>')
            except Error:
                raise Error("Neither <Dim> nor <InputDim> were found")
        in_dim = read_binary_integer32_token(pb)

        try:
            collect_until_token(pb, b'<TargetRms>')
            target_rms = read_binary_float_token(pb)
        except Error:
            # model does not contain TargetRms
            target_rms = 1.0

        try:
            collect_until_token(pb, b'<AddLogStddev>')
            add_log = read_binary_bool_token(pb)
        except Error:
            # model does not contain AddLogStddev
            add_log = False

        if add_log is not False:
            raise Error("AddLogStddev True  in Normalize component is not supported")

        scale = target_rms * np.sqrt(in_dim)

        attrs = {
                 'eps': 0.00000001,
                 'across_spatial': 0,
                 'channel_shared': 1,
                 'in_dim': in_dim,
        }
        embed_input(attrs, 1, 'weights', [scale])

        NormalizeOp.update_node_stat(node, attrs)
        return cls.enabled

# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ctc_greedy_decoder_seq_len import CTCGreedyDecoderSeqLenOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class CTCCGreedyDecoderFrontExtractor(FrontExtractorOp):
    op = 'CTCGreedyDecoder'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'merge_repeated': bool(onnx_attr(node, 'merge_repeated', 'i', default=1)),
        }
        CTCGreedyDecoderSeqLenOp.update_node_stat(node, attrs)
        return cls.enabled

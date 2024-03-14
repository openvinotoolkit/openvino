# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ctc_loss import CTCLoss
from openvino.tools.mo.front.extractor import FrontExtractorOp


class CTCLossFrontExtractor(FrontExtractorOp):
    op = 'CTCLoss'
    enabled = True

    @classmethod
    def extract(cls, node):
        # For CTCLoss default value is [N, T]
        logits_time_major = True
        if 'logits_time_major' in node.pb.attr:
            logits_time_major = node.pb.attr['logits_time_major'].b

        attrs = {
            'ctc_merge_repeated': node.pb.attr['ctc_merge_repeated'].b,
            'preprocess_collapse_repeated': node.pb.attr['preprocess_collapse_repeated'].b,
            'logits_time_major': logits_time_major,
            # unique is always false for CTCLoss V1
            'unique': False
        }

        CTCLoss.update_node_stat(node, attrs)
        return cls.enabled

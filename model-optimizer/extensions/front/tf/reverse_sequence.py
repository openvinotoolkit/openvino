# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.extractor import FrontExtractorOp


class ReverseSequenceFrontExtractor(FrontExtractorOp):
    op = 'ReverseSequence'
    enabled = True

    @classmethod
    def extract(cls, node):
        if node.has_valid('seq_dim'):
            return

        ReverseSequence.update_node_stat(node, {
            'seq_axis': node.pb.attr['seq_dim'].i,
            'batch_axis': node.pb.attr['batch_dim'].i,
        })
        return cls.enabled

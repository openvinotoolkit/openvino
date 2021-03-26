# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.BlockLSTM import BlockLSTM
from mo.front.extractor import FrontExtractorOp


class BlockLSTMExtractor(FrontExtractorOp):
    op = 'BlockLSTM'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'use_peephole': node.pb.attr['use_peephole'].b,
            'cell_clip': node.pb.attr['cell_clip'].f,
            'forget_bias': node.pb.attr['forget_bias'].f,
        }
        BlockLSTM.update_node_stat(node, attrs)
        return cls.enabled

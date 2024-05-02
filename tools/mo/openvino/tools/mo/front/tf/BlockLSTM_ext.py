# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.BlockLSTM import BlockLSTM


class BlockLSTMExtractor(FrontExtractorOp):
    op = "BlockLSTM"
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            "use_peephole": node.pb.attr["use_peephole"].b,
            "cell_clip": node.pb.attr["cell_clip"].f,
            "forget_bias": node.pb.attr["forget_bias"].f,
        }
        BlockLSTM.update_node_stat(node, attrs)
        return cls.enabled

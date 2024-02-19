# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.memoryoffset import MemoryOffset


class MemoryOffsetFrontExtractor(FrontExtractorOp):
    op = 'MemoryOffset'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters
        mapping_rule = {
            'pair_name': pb['pair_name'],
            't': pb['t'],
            'has_default': pb['has_default'],
            'splitted': False,
        }
        if 'element_size' in pb:
            mapping_rule['element_size'] = pb['element_size']

        MemoryOffset.update_node_stat(node, mapping_rule)
        return cls.enabled

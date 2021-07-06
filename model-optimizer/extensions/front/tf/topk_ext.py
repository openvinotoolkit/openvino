# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.topk import TopK
from mo.front.extractor import FrontExtractorOp


class TopKExtractor(FrontExtractorOp):
    op = 'TopK'
    enabled = True

    @classmethod
    def extract(cls, node):
        sort = 'value' if node.pb.attr['sorted'] else 'none'
        TopK.update_node_stat(node, {'mode': 'max', 'axis': -1, 'sort': sort, 'k': node.pb.attr['k'].i,
                                     'index_element_type': np.int32})

        return cls.enabled


class TopKV2Extractor(FrontExtractorOp):
    op = 'TopKV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        sort = 'value' if node.pb.attr['sorted'] else 'none'
        TopK.update_node_stat(node, {'mode': 'max', 'axis': -1, 'sort': sort, 'index_element_type': np.int32})
        return cls.enabled

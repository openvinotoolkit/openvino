# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.ops.op import Op


class ExperimentalDetectronGenerateProposalsSingleImage(Op):
    op = 'ExperimentalDetectronGenerateProposalsSingleImage'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            version='experimental',
            infer=__class__.infer
        )

        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return [
            'min_size',
            'nms_threshold',
            'post_nms_count',
            'pre_nms_count'
        ]

    @staticmethod
    def infer(node):
        node.out_node(0).shape = np.array([node.post_nms_count, 4], dtype=np.int64)
        node.out_node(1).shape = np.array([node.post_nms_count], dtype=np.int64)

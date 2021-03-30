# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.ops.op import Op


class ExperimentalDetectronTopKROIs(Op):
    op = 'ExperimentalDetectronTopKROIs'

    def __init__(self, graph, attrs):
        mandatory_props = dict(
            type=__class__.op,
            op=__class__.op,
            version='experimental',
            infer=__class__.infer
        )
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['max_rois', ]

    @staticmethod
    def infer(node):
        node.out_node(0).shape = np.array([node.max_rois, 4], dtype=np.int64)

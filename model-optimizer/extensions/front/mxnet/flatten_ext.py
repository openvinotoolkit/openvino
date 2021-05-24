# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.ops.flatten import Flatten


class FlattenFrontExtractor(FrontExtractorOp):
    op = 'Flatten'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'axis': 1,
            'end_axis': -1,
        }

        Flatten.update_node_stat(node, attrs)
        return cls.enabled

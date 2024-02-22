# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.unique import Unique
from openvino.tools.mo.front.extractor import FrontExtractorOp


class UniqueFrontExtractor(FrontExtractorOp):
    op = 'Unique'
    enabled = True

    @classmethod
    def extract(cls, node):
        # TensorFlow Unique operation always returns two outputs: unique elements and indices
        # The unique elements in the output are not sorted
        attrs = {
            'sorted': 'false',
            'return_inverse': 'true',
            'return_counts': 'false'
        }

        Unique.update_node_stat(node, attrs)

        return cls.enabled

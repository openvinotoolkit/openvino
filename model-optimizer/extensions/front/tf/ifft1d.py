# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.TFFFT import TFFFT
from mo.front.extractor import FrontExtractorOp


class IFFT1DFrontExtractor(FrontExtractorOp):
    op = 'IFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'is_inverse': True}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled

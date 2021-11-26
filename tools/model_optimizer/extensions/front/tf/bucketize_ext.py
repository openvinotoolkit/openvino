# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.bucketize import Bucketize
from mo.front.extractor import FrontExtractorOp


class BucketizeFrontExtractor(FrontExtractorOp):
    op = 'Bucketize'
    enabled = True

    @classmethod
    def extract(cls, node):
        boundaries = np.array(node.pb.attr['boundaries'].list.f, dtype=np.float)
        Bucketize.update_node_stat(node, {'boundaries': boundaries, 'with_right_bound': False, 'output_type': np.int32})
        return cls.enabled

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.bucketize import Bucketize
from openvino.tools.mo.front.extractor import FrontExtractorOp


class BucketizeFrontExtractor(FrontExtractorOp):
    op = 'Bucketize'
    enabled = True

    @classmethod
    def extract(cls, node):
        boundaries = mo_array(node.pb.attr['boundaries'].list.f, dtype=np.float32)
        Bucketize.update_node_stat(node, {'boundaries': boundaries, 'with_right_bound': False, 'output_type': np.int32})
        return cls.enabled

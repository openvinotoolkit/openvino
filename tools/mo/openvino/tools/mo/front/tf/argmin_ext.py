# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.argmin import ArgMinOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor


class ArgMinFrontExtractor(FrontExtractorOp):
    op = 'ArgMin'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'top_k': 1,
            'axis': None,
            'keepdims': 0,
            'remove_values_output': True,
            'output_type': tf_dtype_extractor(node.pb.attr['output_type'].type, np.int64)
        }
        ArgMinOp.update_node_stat(node, attrs)
        return cls.enabled

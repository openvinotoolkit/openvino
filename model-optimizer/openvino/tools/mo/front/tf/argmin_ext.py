# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.argmin import ArgMinOp
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.extractors.utils import tf_dtype_extractor


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

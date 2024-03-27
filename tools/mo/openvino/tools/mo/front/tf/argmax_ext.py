# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.argmax import ArgMaxOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor


class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @classmethod
    def extract(cls, node):
        ArgMaxOp.update_node_stat(node, {'out_max_val': 0, 'top_k': 1, 'axis': None,
                                         'dim_attrs': ['axis'], 'keepdims': 0, 'remove_values_output': True,
                                         'output_type': tf_dtype_extractor(node.pb.attr['output_type'].type, np.int64),
                                         })
        return cls.enabled

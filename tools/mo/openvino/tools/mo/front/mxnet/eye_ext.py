# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.eye import MXEye
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class EyeExtractor(FrontExtractorOp):
    op = '_eye'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        num_rows = attrs.int("N")
        num_columns = attrs.int("M", num_rows)
        if num_columns is None or num_columns == 0:
            num_columns = num_rows
        diagonal_index = attrs.int("k", 0)
        out_type = attrs.dtype("dtype", np.float32)
        new_attrs = {'num_rows': num_rows, 'num_columns': num_columns, 'diagonal_index': diagonal_index, 'output_type': out_type}
        MXEye.update_node_stat(node, new_attrs)
        return cls.enabled

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.expand_dims import ExpandDims


class UnsqueezeFrontExtractor(FrontExtractorOp):
    """
    Convert Unsqueeze layer to ExpandDims because the ExpandDims layer has fixed attribute with dimensions to unsqueeze.
    """
    op = 'Unsqueeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))

        attrs = {
            'expand_axis': axis if len(axis) != 0 else None
        }

        ExpandDims.update_node_stat(node, attrs)
        return cls.enabled

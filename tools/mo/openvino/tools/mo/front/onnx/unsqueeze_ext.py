# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from openvino.tools.mo.ops.expand_dims import ExpandDims


class UnsqueezeFrontExtractor(FrontExtractorOp):
    """
    Convert Unsqueeze layer to ExpandDims because the ExpandDims layer has fixed attribute with dimensions to unsqueeze.
    """
    op = 'Unsqueeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        onnx_opset_version = get_onnx_opset_version(node)
        # since unsqueeze-13 axes is no longer an attribute
        if onnx_opset_version < 13:
            axis = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))
            ExpandDims.update_node_stat(node, {'expand_axis': axis})
        return cls.enabled

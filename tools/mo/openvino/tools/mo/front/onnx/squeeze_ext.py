# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from openvino.tools.mo.ops.squeeze import Squeeze


class SqueezeFrontExtractor(FrontExtractorOp):
    op = 'Squeeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        # since squeeze-13 axes is no longer an attribute
        if get_onnx_opset_version(node) < 13:
            axis = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))
            attrs = {
                'squeeze_dims': axis if len(axis) != 0 else None
            }
            # update the attributes of the node
            Squeeze.update_node_stat(node, attrs)
        return cls.enabled

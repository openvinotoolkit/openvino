# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version, onnx_node_has_attr
from openvino.tools.mo.ops.clamp import Clamp, AttributedClamp


class ClipFrontExtractor(FrontExtractorOp):
    op = 'Clip'
    enabled = True

    @classmethod
    def extract(cls, node):
        if get_onnx_opset_version(node) < 11:
            attrs = {
                'min': onnx_attr(node, 'min', 'f', np.finfo(np.float32).min),
                'max': onnx_attr(node, 'max', 'f', np.finfo(np.float32).max),
            }
            AttributedClamp.update_node_stat(node, attrs)
        else:
            if onnx_node_has_attr(node, 'min') or onnx_node_has_attr(node, 'max'):
                log.error("ONNX Clip-11 operation '{}' shouldn't have attributes 'min' and 'max', this may mean that "
                          "this operation created with older opset version.".format(
                    node.soft_get('name', node.id)), extra={'is_warning': True})
            Clamp.update_node_stat(node)
        return cls.enabled

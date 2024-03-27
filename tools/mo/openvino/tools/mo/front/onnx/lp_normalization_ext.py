# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.ops.normalize_l2 import NormalizeL2Op
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class LPNormalizeExtractor(FrontExtractorOp):
    op = 'LpNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'p': onnx_attr(node, 'p', 'i', 2),
            'axis': onnx_attr(node, 'axis', 'i', -1),
            'eps_mode': 'add',  # TODO check ONNX implementation
            'eps': 1e-6,  # TODO check ONNX implementation
        }
        if attrs['p'] == 1:
            log.debug('The node {} has unsupported attribute "p" = {}'.format(node.soft_get('name'), attrs['p']))
            return False

        NormalizeL2Op.update_node_stat(node, attrs)
        return cls.enabled

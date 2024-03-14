# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class AffineFrontExtractor(FrontExtractorOp):
    # Affine operation will be transformed to ImageScalar and further will be converted to Mul->Add seq
    op = 'Affine'
    enabled = True

    @classmethod
    def extract(cls, node):
        dst_type = lambda x: mo_array(x)

        scale = onnx_attr(node, 'alpha', 'f', default=None, dst_type=dst_type)
        bias = onnx_attr(node, 'beta', 'f', default=None, dst_type=dst_type)

        node['scale'] = scale
        node['bias'] = bias
        node['op'] = 'ImageScaler'

        return cls.enabled

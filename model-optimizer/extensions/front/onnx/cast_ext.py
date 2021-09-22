# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.Cast import Cast
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import get_onnx_datatype_as_numpy, onnx_attr


class CastFrontExtractor(FrontExtractorOp):
    op = 'Cast'
    enabled = True

    @classmethod
    def extract(cls, node):
        to = onnx_attr(node, 'to', 'i', default=None)
        Cast.update_node_stat(node, {'dst_type': get_onnx_datatype_as_numpy(to)})
        return cls.enabled

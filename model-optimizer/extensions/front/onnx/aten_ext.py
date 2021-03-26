# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.aten import ATen
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ATenFrontExtractor(FrontExtractorOp):
    op = 'ATen'
    enabled = True

    @classmethod
    def extract(cls, node):
        mode = onnx_attr(node, 'mode', 'i', default=1)
        operator = onnx_attr(node, 'operator', 's').decode()

        ATen.update_node_stat(node, {'operator': operator, 'mode': mode})
        return cls.enabled

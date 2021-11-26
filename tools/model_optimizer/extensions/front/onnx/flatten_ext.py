# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.flatten import FlattenONNX


class FlattenFrontExtractor(FrontExtractorOp):
    op = 'Flatten'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', 1)
        attrs = {
            'axis': axis
        }

        FlattenONNX.update_node_stat(node, attrs)
        return cls.enabled

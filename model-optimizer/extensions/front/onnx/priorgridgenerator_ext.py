# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.priorgridgenerator_onnx import ExperimentalDetectronPriorGridGenerator
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronPriorGridGeneratorFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronPriorGridGenerator'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = dict(h=onnx_attr(node, 'h', 'i', 0),
                     w=onnx_attr(node, 'w', 'i', 0),
                     stride_x=onnx_attr(node, 'stride_x', 'f', 0),
                     stride_y=onnx_attr(node, 'stride_y', 'f', 0),
                     flatten=onnx_attr(node, 'flatten', 'i', 1)
                     )
        ExperimentalDetectronPriorGridGenerator.update_node_stat(node, attrs)
        return cls.enabled

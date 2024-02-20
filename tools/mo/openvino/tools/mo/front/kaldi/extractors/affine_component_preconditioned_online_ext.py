# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.extractors.fixed_affine_component_ext import FixedAffineComponentFrontExtractor
from openvino.tools.mo.front.kaldi.utils import read_learning_info
from openvino.tools.mo.graph.graph import Node


class AffineComponentFrontExtractor(FrontExtractorOp):
    op = 'affinecomponentpreconditionedonline'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        read_learning_info(node.parameters)
        return FixedAffineComponentFrontExtractor.extract(node)

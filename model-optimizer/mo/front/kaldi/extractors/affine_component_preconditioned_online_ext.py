# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.extractors.fixed_affine_component_ext import FixedAffineComponentFrontExtractor
from mo.front.kaldi.utils import read_learning_info
from mo.graph.graph import Node


class AffineComponentFrontExtractor(FrontExtractorOp):
    op = 'affinecomponentpreconditionedonline'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        read_learning_info(node.parameters)
        return FixedAffineComponentFrontExtractor.extract(node)

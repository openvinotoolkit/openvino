# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.kaldi.extractors.fixed_affine_component_ext import FixedAffineComponentFrontExtractor
from mo.front.kaldi.utils import read_learning_info


class NaturalGradientAffineComponentFrontExtractor(FrontExtractorOp):
    op = 'naturalgradientaffinecomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        read_learning_info(node.parameters)
        return FixedAffineComponentFrontExtractor.extract(node)

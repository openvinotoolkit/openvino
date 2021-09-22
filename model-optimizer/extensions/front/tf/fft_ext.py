# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.TFFFT import TFFFT
from mo.front.extractor import FrontExtractorOp


class FFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'is_inverse': False}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class FFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'is_inverse': False}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class FFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'is_inverse': False}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'is_inverse': True}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'is_inverse': True}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'is_inverse': True}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled

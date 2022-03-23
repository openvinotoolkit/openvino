# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.TFFFT import TFFFT
from openvino.tools.mo.ops.TFIRFFT import TFIRFFT
from openvino.tools.mo.ops.TFRFFT import TFRFFT
from openvino.tools.mo.front.extractor import FrontExtractorOp


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


class RFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1}
        TFRFFT.update_node_stat(node, attrs)
        return cls.enabled


class RFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2}
        TFRFFT.update_node_stat(node, attrs)
        return cls.enabled


class RFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3}
        TFRFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1}
        TFIRFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2}
        TFIRFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3}
        TFIRFFT.update_node_stat(node, attrs)
        return cls.enabled

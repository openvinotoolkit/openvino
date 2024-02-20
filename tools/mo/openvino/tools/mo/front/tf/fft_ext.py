# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.TFFFT import TFFFT


class FFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'fft_kind': 'DFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class FFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'fft_kind': 'DFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class FFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'FFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'fft_kind': 'DFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'fft_kind': 'IDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'fft_kind': 'IDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'IFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'fft_kind': 'IDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class RFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'fft_kind': 'RDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class RFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'fft_kind': 'RDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class RFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'RFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'fft_kind': 'RDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT1DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 1, 'fft_kind': 'IRDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT2DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 2, 'fft_kind': 'IRDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled


class IRFFT3DOpFrontExtractor(FrontExtractorOp):
    op = 'IRFFT3D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {'num_of_dimensions': 3, 'fft_kind': 'IRDFT'}
        TFFFT.update_node_stat(node, attrs)
        return cls.enabled

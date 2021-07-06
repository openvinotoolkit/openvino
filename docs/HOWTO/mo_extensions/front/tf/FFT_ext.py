# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ! [fft_ext:extractor]
from ...ops.FFT import FFT
from mo.front.extractor import FrontExtractorOp
from mo.utils.error import Error


class FFT2DFrontExtractor(FrontExtractorOp):
    op = 'FFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 0
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled


class IFFT2DFrontExtractor(FrontExtractorOp):
    op = 'IFFT2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'inverse': 1
        }
        FFT.update_node_stat(node, attrs)
        return cls.enabled
# ! [fft_ext:extractor]

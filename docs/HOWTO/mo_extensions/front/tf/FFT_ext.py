# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ! [fft_ext:extractor]
from ...ops.FFT import FFT
from openvino.tools.mo.front.extractor import FrontExtractorOp


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

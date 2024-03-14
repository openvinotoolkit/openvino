# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.mxfft import MXFFT
from openvino.tools.mo.front.extractor import FrontExtractorOp


class FFTFrontExtractor(FrontExtractorOp):
    op = 'fft'
    enabled = True

    @classmethod
    def extract(cls, node):
        MXFFT.update_node_stat(node, {'is_inverse': False})
        return cls.enabled


class IFFTFrontExtractor(FrontExtractorOp):
    op = 'ifft'
    enabled = True

    @classmethod
    def extract(cls, node):
        MXFFT.update_node_stat(node, {'is_inverse': True})
        return cls.enabled

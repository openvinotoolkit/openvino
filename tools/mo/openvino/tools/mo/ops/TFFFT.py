# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class TFFFT(Op):
    """
    This operation is intended to read TF operations FFT, FFT2D, FFT3D, IFFT, IFFT2D, IFFT3D, RFFT, RFFT2D, RFFT3D,
    IRFFT, IRFFT2D, IRFFT3D. The operation TFFFT has two attributes: an integer attribute num_of_dimensions and
    a string attribute fft_kind.

    If an operation is used to read FFT, FFT2D, or FFT3D, then the attribute 'fft_kind' is 'DFT'.
    If an operation is used to read IFFT, IFFT2D, or IFFT3D, then the attribute 'fft_kind' is 'IDFT'.
    If an operation is used to read RFFT, RFFT2D, or RFFT3D, then the attribute 'fft_kind' is 'RDFT'.
    If an operation is used to read IRFFT, IRFFT2D, or IRFFT3D, then the attribute 'fft_kind' is 'IRDFT'.

    The attribute 'num_of_dimensions' is equal to number of transformed axes, i.e. 1 for FFT, IFFT, RFFT, and IRFFT;
    2 for FFT2D, IFFT2D, RFFT2D, and IRFFT2D; 3 for FFT3D, IFFT3D, RFFT3D, and IRFFT3D.

    The transformation TFFFTToDFT converts the operation TFFFT into MO operation according to the following rules:
        1) FFT, FFT2D, FFT3D are converted into DFT;
        2) IFFT, IFFT2D, IFFT3D are converted into IDFT;
        3) RFFT, RFFT2D, RFFT3D are converted into RDFT;
        4) IRFFT, IRFFT2D, IRFFT3D are converted into IRDFT.
    """
    op = 'TFFFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 1,
        }
        assert 'fft_kind' in attrs, 'Attribute fft_kind is not given for the operation TFFFT.'
        assert 'num_of_dimensions' in attrs, 'Attribute num_of_dimensions is not given for the operation TFFFT.'
        super().__init__(graph, mandatory_props, attrs)

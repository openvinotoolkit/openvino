# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.op import Op


class MXFFT(Op):
    """
    This operation is intended to read MxNet operations FFT and IFFT.
    The operation MxMetFFT has one attribute: a boolean attribute is_inverse.

    If an operation to read is FFT, then the attribute 'is_inverse' is False, and True otherwise.

    The transformation MxNetFFTToDFT converts the operation MxNetFFT into MO DFT (if the attribute 'is_inverse'
    is False), or into MO IDFT (otherwise).
    """
    op = 'MXFFT'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

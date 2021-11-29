# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Graph
from mo.ops.op import Op


class TFFFT(Op):
    """
    This operation is intended to read TF operations FFT, FFT2D, FFT3D, IFFT, IFFT2D, IFFT3D.
    The operation TFFFT has two attributes: an integer attribute num_of_dimensions and a boolean attribute is_inverse.

    If an operation to read is FFT, FFT2D, or FFT3D, then the attribute 'is_inverse' is False, and True otherwise.
    The attribute 'num_of_dimensions' is equal to number of transformed axes, i.e. 1 for FFT and IFFT, 2 for FFT2D and
    IFFT2D, 3 for FFT3D and IFFT3D.

    The transformation TFFFTToDFT converts the operation TFFFT into MO DFT (if the attribute 'is_inverse' is False),
    or into MO IDFT (otherwise).
    """
    op = 'TFFFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 1,
        }
        assert 'is_inverse' in attrs, 'Attribute is_inverse is not given for the operation TFFFT.'
        assert 'num_of_dimensions' in attrs, 'Attribute num_of_dimensions is not given for the operation TFFFT.'
        super().__init__(graph, mandatory_props, attrs)

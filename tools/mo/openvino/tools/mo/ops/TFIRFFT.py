# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class TFIRFFT(Op):
    """
    This operation is intended to read TF operations IRFFT, IRFFT2D, IRFFT3D.
    The operation TFIRFFT has one attribute: an integer attribute num_of_dimensions.

    The attribute 'num_of_dimensions' is equal to number of transformed axes, i.e. 1 for IRFFT, 2 for IRFFT2D,
    3 for IRFFT3D.

    The transformation TFIRFFTToIRDFT converts the operation TFIRFFT into MO IRDFT op.
    """
    op = 'TFIRFFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': self.op,
            'out_ports_count': 1,
            'in_ports_count': 2,
        }
        assert 'num_of_dimensions' in attrs, 'Attribute num_of_dimensions is not given for the operation TFIRFFT.'
        super().__init__(graph, mandatory_props, attrs)

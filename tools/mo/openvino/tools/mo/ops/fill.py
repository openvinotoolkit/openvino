# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.op import Op


class Fill(Op):
    """
    The Fill layer tiles the second input tensor (0D constant) to the shape specified in the first input.

    This operation is converted to Broadcast layer.
    """
    op = 'Fill'
    enabled = False

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': None,
            'op': __class__.op,
            'infer': None,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer


class Concat(Op):
    op = 'Concat'
    enabled = True

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': self.op,
            'op': self.op,
            'version': 'opset1',
            'axis': 1,
            'infer': concat_infer,
            'reverse_infer': reverse_bypass_infer,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

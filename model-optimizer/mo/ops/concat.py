# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.concat import concat_infer
from mo.ops.op import Op


class Concat(Op):
    op = 'Concat'
    enabled = True

    def __init__(self, graph, attrs: dict):
        super().__init__(graph, {
            'type': __class__.op,
            'op': __class__.op,
            'version': 'opset1',
            'axis': 1,
            'infer': concat_infer,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return ['axis']

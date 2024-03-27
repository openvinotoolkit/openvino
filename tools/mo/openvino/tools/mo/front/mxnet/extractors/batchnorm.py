# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.batch_norm import batch_norm_4_infer
from openvino.tools.mo.front.common.partial_infer.utils import reverse_bypass_infer

def batch_norm_ext(attrs):
    node_attrs = {
        'type': 'BatchNormalization',
        'eps': attrs.float('eps', 0.001),
        'infer': batch_norm_4_infer,
        'reverse_infer': lambda node: reverse_bypass_infer(node, in_ports=[0]),
        'fix_gamma': attrs.bool('fix_gamma', False)
    }
    node_attrs.update(layout_attrs())
    return node_attrs

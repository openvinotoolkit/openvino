# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.batch_norm import batch_norm_4_infer


def batch_norm_ext(attrs):
    node_attrs = {
        'type': 'BatchNormalization',
        'eps': attrs.float('eps', 0.001),
        'infer': batch_norm_4_infer,
        'reverse_infer': batch_norm_reverse_infer,
        'fix_gamma': attrs.bool('fix_gamma', False)
    }
    node_attrs.update(layout_attrs())
    return node_attrs


def batch_norm_reverse_infer(node):
    output_shape = node.out_port(0).data.get_shape()
    if output_shape is not None and node.in_port(0).data.get_shape() is None:
        node.in_port(0).data.set_shape(output_shape)

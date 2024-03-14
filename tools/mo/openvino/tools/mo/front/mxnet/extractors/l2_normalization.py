# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.elemental import copy_shape_infer


def l2_normalization_ext(attrs):
    eps = attrs.float('eps', 1e-10)

    node_attrs = {
        'op': 'Normalize',
        'type': 'Normalize',
        'eps': eps,
        'across_spatial': 0,
        'channel_shared': 0,
        'infer': copy_shape_infer
    }
    return node_attrs

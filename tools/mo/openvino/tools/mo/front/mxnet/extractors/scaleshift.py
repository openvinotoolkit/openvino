# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.common.partial_infer.batch_norm import batch_norm_4_infer


def scale_shift_ext(attrs):
    node_attrs = {
        'type': 'ScaleShift',
        'fix_gamma': attrs.bool("fix_gamma", True),
        'infer': batch_norm_4_infer
    }
    node_attrs.update(layout_attrs())
    return node_attrs

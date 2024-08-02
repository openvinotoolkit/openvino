# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.multi_box_prior import multi_box_prior_infer_mxnet


def multi_box_prior_ext(attr):
    min_size = list(attr.tuple("sizes", float, (1, 1)))
    offset_y, offset_x = attr.tuple("offsets", float, (0.5, 0.5))
    clip = 0 if not attr.bool("clip", False) else 1
    aspect_ratio = attr.tuple("ratios", float, None)
    step_y, step_x = attr.tuple("steps", float, (-1, -1))
    if len(aspect_ratio) == 0:
        aspect_ratio = [1.0]

    node_attrs = {
        'type': 'PriorBox',
        'step': step_x,
        'offset': offset_x,
        'variance': '0.100000,0.100000,0.200000,0.200000',
        'flip': 0,
        'clip': clip,
        'min_size': min_size,
        'max_size': '',
        'aspect_ratio': list(aspect_ratio),
        'scale_all_sizes': 0,
        'infer': multi_box_prior_infer_mxnet
    }
    return node_attrs

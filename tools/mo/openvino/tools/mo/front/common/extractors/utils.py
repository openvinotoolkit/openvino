# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array


def layout_attrs():
    return {
        'spatial_dims': int64_array([2, 3]),
        'channel_dims': int64_array([1]),
        'batch_dims': int64_array([0]),
        'layout': 'NCHW'
    }

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array


def layout_attrs():
    return {
        'spatial_dims': mo_array([2, 3], dtype=np.int64),
        'channel_dims': mo_array([1], dtype=np.int64),
        'batch_dims': mo_array([0], dtype=np.int64),
        'layout': 'NCHW'
    }

# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def layout_attrs():
    return {
        'spatial_dims': np.array([2, 3], dtype=np.int64),
        'channel_dims': np.array([1], dtype=np.int64),
        'batch_dims': np.array([0], dtype=np.int64),
        'layout': 'NCHW'
    }

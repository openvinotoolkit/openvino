# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import openvino.runtime.opset1 as ov
from tests.test_ngraph.util import run_op_node


def test_sparse_conv():
    features = np.array([[1.0], [1.0]])
    inp_pos = np.array([[1.46057, 3.3381, 0.504631], [1.00087, 2.48036, 1.01154]])
    out_pos = inp_pos
    kernel = np.arange(1, 3 * 3 * 3 + 1).reshape(1, 1, 3, 3, 3).astype(np.float32)
    offset = np.array([0, 0, 0]).astype(np.float32)
    voxel_size = np.array([1.0])
    result = run_op_node([features, inp_pos, out_pos, kernel, offset, voxel_size], ov.sparse_conv)[0]
    assert np.allclose(
        result,
        np.array([ [34.0], [22.0] ], dtype=np.float32),
    )

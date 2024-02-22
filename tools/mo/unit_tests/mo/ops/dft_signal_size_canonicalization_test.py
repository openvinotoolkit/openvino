# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.dft import FFTBase
from openvino.tools.mo.front.common.partial_infer.utils import int64_array


class TestDFTSignalSizeCanonicalizationTest():
    @pytest.mark.parametrize("signal_size, axes, input_shape, expected_result",[
        (int64_array([-1, 77]), int64_array([1, 2]), int64_array([2, 180, 180, 2]), int64_array([180, 77])),
        (int64_array([390, 87]), int64_array([2, 0]), int64_array([2, 180, 180, 2]), int64_array([390, 87])),
        (int64_array([600, -1, 40]),
         int64_array([3, 0, 1]),
         int64_array([7, 50, 130, 400, 2]),
         int64_array([600, 7, 40])),
        (int64_array([-1, 16, -1]),
         int64_array([3, 0, 2]),
         int64_array([7, 50, 130, 400, 2]),
         int64_array([400, 16, 130])),
        (int64_array([16, -1, -1]),
         int64_array([3, 0, 2]),
         int64_array([7, 50, 130, 400, 2]),
         int64_array([16, 7, 130])),
        (int64_array([-1, -1, 16]),
         int64_array([3, 0, 2]),
         int64_array([7, 50, 130, 400, 2]),
         int64_array([400, 7, 16])),
        (int64_array([-1, -1, -1]),
         int64_array([3, 0, 2]),
         int64_array([7, 50, 130, 400, 2]),
         int64_array([400, 7, 130])),
    ])
    def test_canonicalization(self, signal_size, axes, input_shape, expected_result):
        canonicalized_signal_size = FFTBase.canonicalize_signal_size(signal_size, axes, input_shape)
        assert np.array_equal(canonicalized_signal_size, expected_result)

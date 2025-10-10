# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_ravel(torch.nn.Module):
    def __init__(self, complex_type):
        super().__init__()
        self.complex_type = complex_type

    def forward(self, x):
        if self.complex_type:
            x = torch.view_as_complex(x)

        result = torch.ravel(x)

        if self.complex_type:
            result = torch.view_as_real(result)

        return result


class TestRavel(PytorchLayerTest):
    def _prepare_input(self, shape, dtype, complex_type=False):
        if complex_type:
            input_shape = (*shape, 2)
        else:
            input_shape = shape

        data = np.random.uniform(0, 50, input_shape).astype(dtype)
        return (data,)

    def create_model(self, complex_type=False):
        return aten_ravel(complex_type)

    @pytest.mark.parametrize("shape", [
        (5,),
        (2, 3),
        (4, 1, 5),
        (12, 12, 24, 1),
        (24, 1, 12, 1),
        (),
    ])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel(self, shape, dtype, ie_device, precision, ir_version):
        self._test(self.create_model(),
                   None, "aten::ravel", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "shape": shape,
                       "dtype": dtype,
                   })

    @pytest.mark.parametrize("shape", [
        (5,),
        (2, 3),
        (4, 1, 5),
        (12, 12, 24, 1),
        (24, 1, 12, 1),
        (),
    ])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel_complex(self, shape, dtype, ie_device, precision, ir_version):
        self._test(self.create_model(complex_type=True),
                   None, "aten::ravel", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "shape": shape,
                       "dtype": dtype,
                       "complex_type": True
                   })

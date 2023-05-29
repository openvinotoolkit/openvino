# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestUnflatten(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (np.random.uniform(0, 50, (6, 3, 4)).astype(dtype),)

    def create_model(self, dim, shape):
        import torch

        class aten_unflatten(torch.nn.Module):
            def __init__(self, dim, shape):
                super(aten_unflatten, self).__init__()
                self.dim = dim
                self.shape = shape

            def forward(self, x):
                return x.unflatten(self.dim, self.shape)

        ref_net = None

        return aten_unflatten(dim, shape), ref_net, "aten::unflatten"

    @pytest.mark.parametrize(("dim", "shape"), [(0, [2, 1, 3]),  (1, [1, 3]), (2, (2, -1)), (-1, (2, 2)), (-2, (-1, 1))])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_unflatten(self, dim, shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, shape), ie_device, precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})
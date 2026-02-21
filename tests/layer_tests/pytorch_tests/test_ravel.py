# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRavel(PytorchLayerTest):
    def _prepare_input(self, shape, dtype="float32"):
        return (np.random.randn(*shape).astype(dtype),)

    def create_model(self):
        import torch

        class aten_ravel(torch.nn.Module):
            def forward(self, x):
                return torch.ravel(x)

        ref_net = None
        return aten_ravel(), ref_net, "aten::ravel"

    @pytest.mark.parametrize("shape", [
        [2, 3],
        [2, 3, 4],
        [2, 3, 4, 5],
        [1],
        [5],
        [1, 1, 1],
    ])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_ravel(self, shape, dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"shape": shape, "dtype": dtype},
        )
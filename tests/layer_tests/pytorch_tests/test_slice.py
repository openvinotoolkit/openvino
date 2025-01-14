# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestSlice1D(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(range(16), np.float32), np.array(self.params, dtype=np.int32))

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x, params):
                return x[params[0] : params[1] : params[2]]

        ref_net = None

        return aten_slice(), ref_net, "aten::slice"

    @pytest.mark.parametrize(
        "params",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice1d(self, ie_device, precision, ir_version, params):
        self.params = params
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
        )


class TestSlice2D(PytorchLayerTest):
    def _prepare_input(self):
        return (
            np.array([range(16), range(16, 32)], np.float32),
            np.array(self.params_0a, dtype=np.int32),
            np.array(self.params_1a, dtype=np.int32),
        )

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x, params_0a, params_1a):
                return x[params_0a[0] : params_0a[1] : params_0a[2], params_1a[0] : params_1a[1] : params_1a[2]]

        ref_net = None

        return aten_slice(), ref_net, "aten::slice"

    @pytest.mark.parametrize(
        "params_0a",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.parametrize(
        "params_1a",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice2d(self, ie_device, precision, ir_version, params_0a, params_1a):
        self.params_0a = params_0a
        self.params_1a = params_1a
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
        )


class TestSliceAndSqueeze(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 1, 32).astype(np.float32),)

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                a = torch.squeeze(x, 1)
                return a[:, None, :]

        ref_net = None

        return aten_slice(), ref_net, "aten::slice"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice_and_squeeze(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, dynamic_shapes=False)

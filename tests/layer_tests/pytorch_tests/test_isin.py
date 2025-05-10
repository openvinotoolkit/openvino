# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class aten_isin(torch.nn.Module):
    def __init__(self, invert):
        super().__init__()
        self.invert = invert

    def forward(self, x, y):
        return torch.isin(x, y, invert=self.invert)


class TestIsin(PytorchLayerTest):
    def _prepare_input(self, in_type_idx, inputs_idx, scalar):
        rng = np.random.default_rng(12345)
        inputs = [
            (np.concatenate([np.ones(2), rng.random(5)]), np.ones([20, 20])),
            (rng.integers(-10, 10, (10,)), rng.integers(-10, 10, (10, 10))),
            (rng.integers(-10, 10, (10, 5)), rng.integers(-10, 10, (3, 10, 3))),
        ][inputs_idx]
        in_type = [np.float32, np.int32, np.int16, np.float16, np.int8][in_type_idx]
        elements = inputs[0].astype(in_type)
        test_elements = inputs[1].astype(in_type)
        if scalar == 0:
            elements = np.ones(1).astype(in_type)
        elif scalar == 1:
            test_elements = np.ones(1).astype(in_type)
        return elements, test_elements

    @pytest.mark.parametrize("invert", [False, True])
    @pytest.mark.parametrize("in_type_idx", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("scalar", [0, 1, -1])
    @pytest.mark.parametrize("inputs_idx", [0, 1, 2])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_isin(
        self, invert, in_type_idx, inputs_idx, scalar, ie_device, precision, ir_version
    ):
        self._test(
            aten_isin(invert),
            None,
            "aten::isin",
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "in_type_idx": in_type_idx,
                "inputs_idx": inputs_idx,
                "scalar": scalar,
            },
        )

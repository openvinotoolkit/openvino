# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestPoisson(PytorchLayerTest):
    def _prepare_input(self, rates, input_type):
        return [rates.astype(input_type)]

    def create_model(self, seed):
        class aten_poisson(torch.nn.Module):
            def __init__(self, seed):
                super().__init__()
                self.gen = torch.Generator()
                self.seed = seed

            def forward(self, rates):
                self.gen.manual_seed(self.seed)
                return torch.poisson(rates, generator=self.gen)

        ref_net = None
        return aten_poisson(seed), ref_net, "aten::poisson"

    @pytest.mark.parametrize("rates", [
        np.array([1.0, 2.0, 3.0, 5.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([0.5, 1.5, 2.5]),
    ])
    @pytest.mark.parametrize("input_type", [np.float32, np.float64])
    @pytest.mark.parametrize("seed", [1, 50, 1234])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_poisson(self, rates, input_type, seed, ie_device, precision, ir_version):
        self._test(*self.create_model(seed),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"rates": rates, "input_type": input_type})
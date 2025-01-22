# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestBernoulli(PytorchLayerTest):
    def _prepare_input(self, input, input_type, out):
        model_inputs = [input.astype(input_type)]
        if out:
            model_inputs.append(np.zeros_like(input).astype(np.int64))
        return model_inputs

    def create_model(self, out, seed):
        class aten_bernoulli(torch.nn.Module):
            def __init__(self, out, seed) -> None:
                super().__init__()
                gen = torch.Generator()
                gen.manual_seed(seed)
                self.gen = gen
                if not out:
                    self.forward = self.bernoulli
                else:
                    self.forward = self.bernoulli_out

            def bernoulli(self, input):
                bernoulli_res = torch.bernoulli(input, generator=self.gen)
                return bernoulli_res

            def bernoulli_out(self, input, out):
                bernoulli_res = torch.bernoulli(input, generator=self.gen, out=out)
                return bernoulli_res

        ref_net = None

        return aten_bernoulli(out, seed), ref_net, "aten::bernoulli"

    @pytest.mark.parametrize("input", [
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]),
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ])
    @pytest.mark.parametrize("input_type", [np.float32, np.float64])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("seed", [1, 50, 1234])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bernoulli(self, input, input_type, out, seed, ie_device, precision, ir_version):
        if input_type == np.float64:
            pytest.skip("156027: Incorrect specification or reference for RandomUniform for fp64 output type")
        self._test(*self.create_model(out, seed),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input": input,
                                            "input_type": input_type,
                                            "out": out})


class TestBernoulliWithP(PytorchLayerTest):
    def _prepare_input(self, input, input_type):
        model_inputs = [input.astype(input_type)]
        return model_inputs

    def create_model(self, p, seed):
        class aten_bernoulli(torch.nn.Module):
            def __init__(self, p, seed) -> None:
                super().__init__()
                gen = torch.Generator()
                gen.manual_seed(seed)
                self.gen = gen
                self.p = p
                self.forward = self.bernoulli_with_p

            def bernoulli_with_p(self, input):
                bernoulli_res = torch.bernoulli(input, self.p, generator=self.gen)
                return bernoulli_res

        ref_net = None

        return aten_bernoulli(p, seed), ref_net, "aten::bernoulli"

    @pytest.mark.parametrize("input", [
        np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]),
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ])
    @pytest.mark.parametrize("input_type", [np.float32, np.int32, np.float64])
    @pytest.mark.parametrize("p", [0.0, 0.4, 1.0])
    @pytest.mark.parametrize("seed", [12])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bernoulli(self, input, input_type, p, seed, ie_device, precision, ir_version):
        if p not in [0.0, 1.0]:
            pytest.skip("156027: Incorrect specification or reference for RandomUniform for fp64 output type")
        self._test(*self.create_model(p, seed),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input": input,
                                            "input_type": input_type})

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestInstanceNorm(PytorchLayerTest):
    def _prepare_input(self, ndim=4):
        shape5d = [3, 6, 10, 5, 2]
        shape = shape5d[:ndim]
        return (self.random.randn(*shape),)

    def create_model(self, weights=False, bias=False, mean_var=False, use_input_stats=None, eps=1e-05):
        import torch

        class aten_instance_norm(torch.nn.Module):
            def __init__(self, rng, weights=False, bias=False, mean_var=False, use_input_stats=None, eps=1e-05):
                super().__init__()
                weights_shape = (6, )
                self.weight = rng.torch_randn(*weights_shape) if weights else None
                self.bias = None
                # By default normalize with running statistics only when they are provided;
                # use_input_stats can be forced independently to exercise the corners below.
                self.use_input_stats = (not mean_var) if use_input_stats is None else use_input_stats
                if bias:
                    self.bias = rng.torch_randn(*weights_shape)
                self.mean = None
                self.var = None
                if mean_var:
                    self.mean = rng.torch_randn(*weights_shape)
                    self.var = rng.torch_randn(*weights_shape)

                self.eps = eps

            def forward(self, x):
                return torch.instance_norm(x, self.weight, self.bias, self.mean, self.var,  self.use_input_stats, 0.1, self.eps, False)


        return aten_instance_norm(self.random, weights, bias, mean_var, use_input_stats, eps), "aten::instance_norm"

    @pytest.mark.parametrize("params",
                             [
                                 {"eps": 0.0001},
                                 {'weights': True, 'eps': -0.05},
                                 {'weights': True},
                                 {'weights': True, 'bias': True},
                                 {"weights": True, 'bias': False, "mean_var": True},
                                 {"weights": True, 'bias': True, "mean_var": True},
                                 {"weights": False, 'bias': True, "mean_var": True},
                                 {"weights": False, 'bias': False, "mean_var": True},
                                 {"weights": False, 'bias': False,
                                  "mean_var": True, "eps": 1.5},
                                 # use_input_stats=True WITH running stats provided -> running stats must be IGNORED
                                 {"weights": True, 'bias': True, "mean_var": True, "use_input_stats": True},
                                 # use_input_stats=False WITH running stats provided -> running stats used
                                 {"weights": True, 'bias': True, "mean_var": True, "use_input_stats": False},
                                 {"weights": True, 'bias': False, "mean_var": True, "use_input_stats": True}
                             ])
    @pytest.mark.parametrize("kwargs_to_prepare_input", [
        {"ndim": 3},
        {'ndim': 4},
        {"ndim": 5}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_group_norm(self, params, ie_device, precision, ir_version, kwargs_to_prepare_input):
        self._test(*self.create_model(**params),
                   ie_device, precision, ir_version, kwargs_to_prepare_input=kwargs_to_prepare_input,
                   dynamic_shapes=not params.get("mean_var", False), use_mo_convert=False)


class TestInstanceNormEvalMode(PytorchLayerTest):
    """use_input_stats=False requires running statistics, as in torch.

    torch raises "Expected running_mean and running_var to be defined when use_input_stats is
    false", so a model asking for the running statistics without providing them must be rejected
    instead of silently normalizing with the input statistics.
    """

    def create_model(self, stats):
        import torch

        class aten_instance_norm(torch.nn.Module):
            def __init__(self, mean, var):
                super().__init__()
                self.mean = mean
                self.var = var

            def forward(self, x):
                return torch.instance_norm(x, None, None, self.mean, self.var, False, 0.1, 1e-05, False)

        weights_shape = (6, )
        mean = self.random.torch_randn(*weights_shape) if stats in ("mean", "both") else None
        var = self.random.torch_randn(*weights_shape) if stats in ("var", "both") else None
        return aten_instance_norm(mean, var), "aten::instance_norm"

    @pytest.mark.parametrize("stats", ["none", "mean", "var"])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_missing_running_stats_fails(self, stats, ie_device, precision, ir_version):
        import torch
        import openvino as ov

        model, _ = self.create_model(stats)
        example = torch.randn(3, 6, 10, 5)
        # tracing runs forward and torch itself rejects the missing statistics, so script the model
        # to hand the graph to the converter as is.
        scripted = torch.jit.script(model)
        with pytest.raises(ov.frontend.OpConversionFailure):
            ov.convert_model(scripted, example_input=(example, ))

    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_running_stats_used(self, ie_device, precision, ir_version):
        import torch
        import openvino as ov

        model, _ = self.create_model("both")
        example = torch.randn(3, 6, 10, 5)
        # control case: with both statistics provided the same scripted model must still convert
        ov_model = ov.convert_model(torch.jit.script(model), example_input=(example, ))
        assert ov_model is not None

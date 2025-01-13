# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestRound(PytorchLayerTest):
    def _prepare_input(self, out=False, dtype="float32"):
        import numpy as np
        input = np.random.randn(1, 3, 224, 224).astype(dtype)
        if dtype == "float64":
            # fp64 can fail by accuracy, because pytorch rounds fp64 value and ov will round fp32 value.
            # To remove sporadic accuracy fails we will round the number to 6 decimal places.
            input = np.round(input, 6)
        if not out:
            return (input, )
        return (input, np.zeros_like(input))

    def create_model(self, out=False):
        import torch

        class aten_round(torch.nn.Module):
            def __init__(self, out):
                super(aten_round, self).__init__()
                if out:
                    self.forward = self.forward_out
                 

            def forward(self, x):
                return torch.round(x)

            def forward_out(self, x, y):
                return torch.round(x, out=y), y

        ref_net = None

        return aten_round(out), ref_net, "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_round(self, out, dtype, ie_device, precision, ir_version):
        if ie_device == "GPU" and dtype not in ["float32", "float64"]:
            pytest.xfail(reason="square is not supported on GPU for integer types")
        self._test(*self.create_model(out), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "dtype": dtype})


class TestRoundScalar(PytorchLayerTest):
    def _prepare_input_int(self):
        import numpy as np
        return (np.array(np.random.randint(low=-5, high=5)), )

    def _prepare_input_float(self):
        import numpy as np
        return (np.array(np.random.uniform(low=-5, high=5)), )

    def create_model(self, input_type="float"):
        import torch

        class aten_round(torch.nn.Module):
            def __init__(self, input_type):
                super(aten_round, self).__init__()
                if input_type == "int":
                    self.forward = self.forward_int
                else:
                    self.forward = self.forward_float
                 

            def forward_int(self, x:int):
                return torch.round(x)

            def forward_float(self, x:float):
                return torch.round(x)

        ref_net = None

        return aten_round(input_type), ref_net, "aten::round"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_type", ["int", "float"])
    def test_round(self, input_type, ie_device, precision, ir_version):
        if input_type == "int":
            if ie_device == "GPU":
                pytest.xfail(reason="round is not supported on GPU for integer types")
            self._prepare_input = self._prepare_input_int
        else:
            self._prepare_input = self._prepare_input_float
        self._test(*self.create_model(input_type), ie_device, precision, ir_version, trace_model=True)

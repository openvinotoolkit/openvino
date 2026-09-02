# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

pytestmark = pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")


class TestInplaceUniform(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, from_val, to_val):
        class aten_uniform(torch.nn.Module):
            def __init__(self, from_val, to_val):
                super().__init__()
                self.from_val = from_val
                self.to_val = to_val

            def forward(self, x):
                x = x.to(torch.float32)
                if self.from_val is None and self.to_val is None:
                    return x.uniform_(), x
                return x.uniform_(self.from_val, self.to_val), x

        return aten_uniform(from_val, to_val), "aten::uniform_"

    @pytest.mark.parametrize("from_val,to_val", [
        (None, None),
        (0.0, 1.0),
        (-5.0, 5.0),
        (10.0, 20.0),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_inplace_uniform(self, from_val, to_val, ie_device, precision, ir_version):
        self._test(*self.create_model(from_val, to_val),
                   ie_device, precision, ir_version, custom_eps=1e30)


class TestUniformStatistics:
    class aten_uniform(torch.nn.Module):
        def __init__(self, from_val, to_val):
            super().__init__()
            self.from_val = from_val
            self.to_val = to_val

        def forward(self, x):
            return x.uniform_(self.from_val, self.to_val)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("from_val,to_val,size", [
        (0.0, 1.0, (100000,)),
        (-5.0, 5.0, (10000, 10)),
        (10.0, 20.0, (100000,)),
    ])
    def test_uniform_statistics(self, from_val, to_val, size, ie_device, precision):
        import numpy.testing as npt
        import openvino as ov

        fw_model = self.aten_uniform(from_val, to_val)
        example_input = (torch.zeros(size, dtype=torch.float32),)
        input_size = [size]

        ov_model = ov.convert_model(input_model=fw_model, example_input=example_input, input=input_size)
        if ie_device == "GPU" and precision == "FP32":
            config = {"INFERENCE_PRECISION_HINT": "f32"}
        else:
            config = {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        ov_res = compiled_model(example_input)[0]

        # Check bounds
        assert np.all(ov_res >= from_val) and np.all(ov_res <= to_val)

        # Check mean property of uniform distribution: mean ≈ (from + to) / 2
        expected_mean = (from_val + to_val) / 2.0
        npt.assert_allclose(np.mean(ov_res), expected_mean, atol=0.1, rtol=0.1)

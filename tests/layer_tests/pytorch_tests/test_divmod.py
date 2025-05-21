# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy.testing as npt
import openvino as ov
import pytest
import torch


# do not test via PytorchLayerTest since PytorchLayerTest triggers own TorchScript tracing
# this test validates TorchScript patched inside ov.convert_model
class TestBuiltinDivmod():
    class divmod_on_assert_path(torch.nn.Module):
        # test divmod() on assert path that is omitted in ov::Model
        def forward(self, x):
            divisor, rem = divmod(x.shape[-1], 7)
            assert rem == 0, x.shape
            assert divisor > 0 and (divisor & (divisor - 1)) == 0, (divisor, x.shape)
            new_shape = tuple(x.shape[:-1]) + (-1, 7)
            return x.reshape(new_shape)

    class divmod_on_compute_path(torch.nn.Module):
        # test divmod() on assert path that is omitted in ov::Model
        def forward(self, x):
            divisor, rem = divmod(x.shape[0], 5)
            divisor = torch.tensor(divisor, dtype=torch.int32)
            rem = torch.tensor(rem, dtype=torch.int32)
            return divisor, rem

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_divmod_on_assert_path(self, ie_device, precision):
        fw_model = self.divmod_on_assert_path()
        inputs = torch.randn(2, 3, 28)

        example_input = inputs
        ov_model = ov.convert_model(input_model=fw_model, example_input=example_input)
        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}
        else:
            config = {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        fw_res = fw_model(example_input)
        ov_res = compiled_model(example_input)[0]

        npt.assert_allclose(fw_res.numpy(), ov_res)

    @pytest.mark.parametrize("x_shape", [[1], [2], [3], [4], [5], [6]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_divmod_on_compute_path(self, ie_device, precision, x_shape):
        fw_model = self.divmod_on_compute_path()
        x = torch.randn(x_shape)

        example_input = x
        ov_model = ov.convert_model(input_model=fw_model,
                                    input=[ov.PartialShape([-1])],
                                    example_input=example_input)

        if ie_device == 'GPU' and precision == 'FP32':
            config = {'INFERENCE_PRECISION_HINT': 'f32'}
        else:
            config = {}
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        fw_div, fw_rem = fw_model(example_input)
        ov_res = compiled_model(example_input)
        ov_div, ov_rem = ov_res[0], ov_res[1]
        npt.assert_allclose(fw_div.numpy(), ov_div)
        npt.assert_allclose(fw_rem.numpy(), ov_rem)

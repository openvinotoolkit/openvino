# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy.testing as npt
import openvino as ov
import os
import pytest
import torch


def get_num_pytest_workers():
    # PYTEST_XDIST_WORKER_COUNT is set by pytest-xdist when running with -n
    return int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))


# do not test via PytorchLayerTest since PytorchLayerTest triggers own TorchScript tracing
# this test validates TorchScript patched inside ov.convert_model
class TestBuiltinDivmod():
    class builtin_divmod(torch.nn.Module):
        # test divmod() on assert path that is omitted in ov::Model
        def forward(self, x):
            divisor, rem = divmod(x.shape[-1], 7)
            assert rem == 0, x.shape
            assert divisor > 0 and (divisor & (divisor - 1)) == 0, (divisor, x.shape)
            new_shape = tuple(x.shape[:-1]) + (-1, 7)
            return x.reshape(new_shape)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("fw_model,inputs", [
        (builtin_divmod(), torch.randn(2, 3, 28)),
    ])
    def test_builtin_divmod(self, fw_model, inputs, ie_device, precision):
        if get_num_pytest_workers() != 1:
            pytest.skip(reason="test passes only for the single worker in pytest.")

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

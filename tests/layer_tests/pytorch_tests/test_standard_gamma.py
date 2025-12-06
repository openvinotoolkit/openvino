# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import openvino as ov


class TestStandardGammaStatistics:
    class AtenStandardGamma(torch.nn.Module):
        def forward(self, alpha):
            return torch._standard_gamma(alpha)

    def _run_gamma_stat_test(
        self,
        alpha_value,
        shape,
        mean_rtol,
        mean_atol,
        var_rtol,
        var_atol,
        ie_device,
        precision,
    ):
        torch.manual_seed(0)
        np.random.seed(0)

        model = self.AtenStandardGamma()
        alpha_np = np.full(shape, alpha_value, dtype=np.float32)
        alpha_tensor = torch.from_numpy(alpha_np)

        ov_model = ov.convert_model(input_model=model, example_input=(alpha_tensor,))
        config = (
            {"INFERENCE_PRECISION_HINT": "f32"}
            if ie_device == "GPU" and precision == "FP32"
            else {}
        )
        compiled_model = ov.Core().compile_model(ov_model, ie_device, config)

        with torch.no_grad():
            fw_samples = model(alpha_tensor).detach().cpu().numpy().reshape(-1)

        infer_request = compiled_model.create_infer_request()
        infer_request.infer({compiled_model.input(0): alpha_np})
        ov_samples = infer_request.get_output_tensor(0).data.reshape(-1)

        assert np.isfinite(fw_samples).all(), "PyTorch gamma samples contain non-finite values"
        assert np.isfinite(ov_samples).all(), "OpenVINO gamma samples contain non-finite values"

        expected_mean = alpha_value
        expected_var = alpha_value

        np.testing.assert_allclose(
            fw_samples.mean(),
            expected_mean,
            rtol=mean_rtol,
            atol=mean_atol,
        )
        np.testing.assert_allclose(
            fw_samples.var(),
            expected_var,
            rtol=var_rtol,
            atol=var_atol,
        )
        np.testing.assert_allclose(
            ov_samples.mean(),
            expected_mean,
            rtol=mean_rtol,
            atol=mean_atol,
        )
        np.testing.assert_allclose(
            ov_samples.var(),
            expected_var,
            rtol=var_rtol,
            atol=var_atol,
        )

    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "alpha_value,shape,mean_rtol,mean_atol,var_rtol,var_atol",
        [
            (0.25, (10_000,), 2e-2, 2e-2, 2e-1, 2e-2),
            (1.0, (10_000,), 2e-2, 2e-2, 2e-1, 2e-2),
        ],
    )
    def test_standard_gamma_statistics_precommit(
        self, alpha_value, shape, mean_rtol, mean_atol, var_rtol, var_atol, ie_device, precision
    ):
        self._run_gamma_stat_test(
            alpha_value, shape, mean_rtol, mean_atol, var_rtol, var_atol, ie_device, precision
        )

    @pytest.mark.nightly
    @pytest.mark.parametrize(
        "alpha_value,shape,mean_rtol,mean_atol,var_rtol,var_atol",
        [
            (0.25, (200_000,), 5e-3, 5e-3, 1e-1, 2e-2),
            (7.5, (50_000,), 1e-2, 1e-2, 1e-1, 2e-2),
        ],
    )
    def test_standard_gamma_statistics_nightly(
        self, alpha_value, shape, mean_rtol, mean_atol, var_rtol, var_atol, ie_device, precision
    ):
        self._run_gamma_stat_test(
            alpha_value, shape, mean_rtol, mean_atol, var_rtol, var_atol, ie_device, precision
        )

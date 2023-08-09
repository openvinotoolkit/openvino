# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_normalize(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x)


class aten_layer_norm(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.layer_norm(x, normalized_shape=[3])


class precision_sensitive_1(torch.nn.Module):
    def forward(self, x):
        eps = 1.0e-8
        return 2.0 / (torch.sqrt(torch.sum(torch.pow(x + 2, 2.0), 1)) + eps)


class precision_sensitive_2(torch.nn.Module):
    def forward(self, x):
        return torch.sum(torch.exp(x + 10), 1)


class precision_sensitive_3(torch.nn.Module):
    def forward(self, x):
        eps = 1.0e-8
        return 2.0 * (torch.sqrt(torch.sum(torch.pow(x + 2, 2.0), 1)) + eps)**(-1)


class precision_sensitive_two_inp_1(torch.nn.Module):
    def forward(self, x, y):
        eps = 1.0e-8
        return x / (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)


class precision_sensitive_two_inp_2(torch.nn.Module):
    def forward(self, x, y):
        eps = 1.0e-8
        return x * (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)**(-1)


class precision_sensitive_with_matmul(torch.nn.Module):
    def forward(self, x, y):
        eps = 1.0e-8
        interm_res = x / (torch.sqrt(torch.sum(torch.pow(y + 2, 2.0), 2)) + eps)
        print(f"interm_res shpae: {interm_res.shape}")
        print(interm_res)
        weights = 1024.0 + torch.zeros(10, 2)
        return torch.mm(interm_res, weights)

class not_precision_sensitive(torch.nn.Module):
    def forward(self, x):
        return torch.sum(x, 1)


class TestPrecisionSensitive(PytorchLayerTest):
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_aten_normalize(self, ie_device, precision, ir_version):
        self._test(aten_normalize(), None, None, ie_device, precision, ir_version, custom_eps=1.0e-3)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="GPU gives incorrect values for MVN")
    def test_layer_norm(self, ie_device, precision, ir_version):
        self._test(aten_layer_norm(), None, None, ie_device, precision, ir_version, custom_eps=1.0e-3)

    def _prepare_input(self):
        import numpy as np
        return 300 + np.random.randn(2, 3).astype(np.float32),

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive(self, ie_device, precision, ir_version):
        self._test(precision_sensitive_1(), None, None, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive_for_exp_reduce(self, ie_device, precision, ir_version):
        self._test(precision_sensitive_2(), None, None, ie_device, precision, ir_version)


class TestPrecisionSensitiveTwoInputs(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return 10000 + np.ones((2, 10), dtype=np.float32), 300 + np.ones((2, 10, 3), dtype=np.float32)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive_two_inputs_1(self, ie_device, precision, ir_version):
        self._test(precision_sensitive_two_inp_1(), None, None, ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive_two_inputs_2(self, ie_device, precision, ir_version):
        self._test(precision_sensitive_two_inp_2(), None, None, ie_device, precision, ir_version)

    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_precision_sensitive_with_matmul(self, ie_device, precision, ir_version):
        # Check that MatMul is not marked as precision sensitive.
        # This test should fail on GPU, overflow happens on MatMul which under no circumstances
        # could be marked as precision sensitive to overcome overflow.
        if ie_device == 'GPU':
            with pytest.raises(Exception):
                self._test(precision_sensitive_with_matmul(), None, None, ie_device, precision, ir_version)
        else:
            self._test(precision_sensitive_with_matmul(), None, None, ie_device, precision, ir_version)


class TestReduceSumNoExp(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return 10000.0 + np.zeros((2, 20), dtype=np.float32),  # 10 000 * 20 = 200 000 > 65504 (fp16_max)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_precision_sensitive_for_exp_reduce(self, ie_device, precision, ir_version):

        # Check that when ReduceSum (when there are no Exp) is not marked as precision sensitive.
        # This test should fail on GPU, overflow happens on ReduceSum which is not
        # marked as precision sensitive if it does not accept values from Exp.
        if ie_device == 'GPU':
            with pytest.raises(Exception):
                self._test(not_precision_sensitive(), None, None, ie_device, precision, ir_version)
        else:
            self._test(not_precision_sensitive(), None, None, ie_device, precision, ir_version)

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestGcd(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_data

    def create_model_tensor_input(self):
        class aten_gcd_tensor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, tensor_a, tensor_b):
                return torch.gcd(tensor_a, tensor_b)

        ref_net = None

        return aten_gcd_tensor(), ref_net, "aten::gcd"

    def create_model_int_input(self):
        class aten_gcd_int(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, int_a: int, int_b: int):
                return torch.tensor(torch.gcd(int_a, int_b))

        ref_net = None

        return aten_gcd_int(), ref_net, "aten::gcd"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_int(self, ie_device, precision, ir_version):
        self.input_data = (np.array(11, dtype=np.int32), np.array(17, dtype=np.int32))
        self._test(
            *self.create_model_int_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_tensor(self, ie_device, precision, ir_version):
        self.input_data = (
            np.array([14, 4, 12, 10, 3, 0], dtype=np.int32),
            np.array([121, 2, 16, 0, 1, 8], dtype=np.int32),
        )
        self._test(
            *self.create_model_tensor_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_int64(self, ie_device, precision, ir_version):
        self.input_data = (np.array(11, dtype=np.int64), np.array(17, dtype=np.int64))
        self._test(
            *self.create_model_int_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_tensor64(self, ie_device, precision, ir_version):
        self.input_data = (
            np.array([14, 4, 12, 10, 3, 0], dtype=np.int64),
            np.array([121, 2, 16, 0, 1, 8], dtype=np.int64),
        )
        self._test(
            *self.create_model_tensor_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_int_diff_dtypes(self, ie_device, precision, ir_version):
        self.input_data = (np.array(11, dtype=np.int64), np.array(17, dtype=np.int32))
        self._test(
            *self.create_model_int_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gcd_tensor_diff_dtypes(self, ie_device, precision, ir_version):
        self.input_data = (
            np.array([14, 4, 12, 10, 3, 0], dtype=np.int64),
            np.array([121, 2, 16, 0, 1, 8], dtype=np.int32),
        )
        self._test(
            *self.create_model_tensor_input(),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
            trace_model=True
        )

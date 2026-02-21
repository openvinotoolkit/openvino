# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from typing import Optional

from pytorch_layer_test_class import PytorchLayerTest


class TestIsIsNotNone(PytorchLayerTest):
    """
    Tests for aten::__is__ and aten::__isnot__ operations.
    
    These ops are used for None identity checks (x is None / x is not None).
    We test by baking the None check into the model at script time,
    since OpenVINO cannot receive None as a runtime input.
    """

    def _prepare_input(self):
        return (np.array([1.0, 2.0, 3.0], dtype=np.float32),)

    def create_model_is_with_none(self):
        """Model where y is always None - tests 'y is None' -> True path"""
        class m(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                y: Optional[torch.Tensor] = None
                if y is None:
                    return x + 1
                return x * 2  # unreachable but needed for TorchScript

        return m(), None, "aten::__is__"

    def create_model_is_with_tensor(self):
        """Model where y is not None - tests 'y is None' -> False path"""
        class m(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                y: Optional[torch.Tensor] = x  # not None
                if y is None:
                    return x + 1  # unreachable
                return x * 2

        return m(), None, "aten::__is__"

    def create_model_isnot_with_none(self):
        """Model where y is always None - tests 'y is not None' -> False path"""
        class m(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                y: Optional[torch.Tensor] = None
                if y is not None:
                    return x * 2  # unreachable
                return x - 1

        return m(), None, "aten::__isnot__"

    def create_model_isnot_with_tensor(self):
        """Model where y is not None - tests 'y is not None' -> True path"""
        class m(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                y: Optional[torch.Tensor] = x
                if y is not None:
                    return x * 2
                return x - 1  # unreachable

        return m(), None, "aten::__isnot__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_is_none_true(self, ie_device, precision, ir_version):
        # y is None, so 'y is None' evaluates to True -> x + 1
        self._test(*self.create_model_is_with_none(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_is_none_false(self, ie_device, precision, ir_version):
        # y is x (not None), so 'y is None' evaluates to False -> x * 2
        self._test(*self.create_model_is_with_tensor(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_isnot_none_false(self, ie_device, precision, ir_version):
        # y is None, so 'y is not None' evaluates to False -> x - 1
        self._test(*self.create_model_isnot_with_none(), ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_isnot_none_true(self, ie_device, precision, ir_version):
        # y is x (not None), so 'y is not None' evaluates to True -> x * 2
        self._test(*self.create_model_isnot_with_tensor(), ie_device, precision, ir_version)

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestReverse(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_tensors

    def create_model(self, num_tensors):
        class aten_reverse(torch.nn.Module):
            def __init__(self, num_tensors):
                super(aten_reverse, self).__init__()
                self.num_tensors = num_tensors

            def forward(self, *args):
                # Create a list and reverse it
                tensor_list = list(args)
                tensor_list.reverse()
                # Return concatenated result to verify order
                return torch.cat(tensor_list, dim=0)

        ref_net = None
        return aten_reverse(num_tensors), ref_net, "aten::reverse"

    @pytest.mark.parametrize("num_tensors", [2, 3, 5])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_reverse(self, num_tensors, ie_device, precision, ir_version):
        """Test aten::reverse which reverses list order"""
        # Create distinct tensors so we can verify reversal
        self.input_tensors = tuple(
            np.full((1, 3), fill_value=i, dtype=np.float32)
            for i in range(num_tensors)
        )

        self._test(
            *self.create_model(num_tensors),
            ie_device, precision, ir_version,
            trace_model=True
        )

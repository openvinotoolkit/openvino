# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_nested_tensor_from_mask(torch.nn.Module):
    def forward(self, tensor, mask):
        return torch._nested_tensor_from_mask(tensor, mask)

class TestNestedTensorFromMask(PytorchLayerTest):
    def _prepare_input(self):

        batch_size = 3
        seq_len = 5
        hidden_dim = 4
        tensor = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
        
        
        mask = np.zeros((batch_size, seq_len), dtype=np.bool_)

        mask[0, :] = True

        mask[1, :3] = True

        mask[2, :4] = True
        
        return (tensor, mask)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_nested_tensor_from_mask(self, ie_device, precision, ir_version):
        model = aten_nested_tensor_from_mask()
        self._test(model, None, "aten::_nested_tensor_from_mask", ie_device, 
                  precision, ir_version)

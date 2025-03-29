# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestDelete(PytorchLayerTest):
    def _prepare_input(self, input_shape, index_val):
        input_tensor = np.random.randn(*input_shape).astype(np.float32)
        index_tensor = np.array(index_val, dtype=np.int32)
        return (input_tensor, index_tensor)

    def create_model(self):
        class aten_delete(torch.nn.Module):
            def forward(self, x, index):
                idx = int(index.item())
                # Delete along axis 0 by concatenating slices before and after idx.
                return torch.cat([x[:idx], x[idx+1:]], dim=0)
        ref_net = None
        return aten_delete(), ref_net, ["aten::slice", "aten::cat"]

    @pytest.mark.parametrize("input_shape,index_val", [
        ((10,), 5),
        ((10,), 0),
        ((10,), 9),
        ((5, 4), 1),
        ((5, 4), 4),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_delete(self, input_shape, index_val, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "index_val": index_val
                   })

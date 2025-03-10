# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexTensor(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, indices_list, safe: bool):
        import torch

        class aten_index_tensor(torch.nn.Module):
            def __init__(self, indices_list):
                super(aten_index_tensor, self).__init__()
                self.indices_list = indices_list

            def forward(self, x):
                if safe:
                    return torch.ops.aten.index.Tensor(x, self.indices_list)
                return torch.ops.aten._unsafe_index.Tensor(x, self.indices_list)

        ref_net = None

        adjusted_indices_list = []
        for indices in indices_list:
            if indices is not None:
                adjusted_indices_list.append(torch.tensor(indices, dtype=torch.int32))
                continue
            adjusted_indices_list.append(None)

        return aten_index_tensor(adjusted_indices_list), ref_net, None

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize('safe', [True, False])
    @pytest.mark.parametrize(('input_shape', 'indices_list'), [
        ([3, 7], [[0], [5, 3, 0]]),
        ([3, 7, 6], [[0], None, None]),
        ([3, 7, 6], [[0], None, [5, 0, 3]]),
        ([3, 7, 6], [[0, 2, 1], None, [5, 0, 3]]),
        ([3, 7, 6], [[0, 2, 1], [4], [5, 0, 3]]),
    ])
    def test_index_tensor(self, safe, input_shape, indices_list, ie_device, precision, ir_version):
        if not PytorchLayerTest.use_torch_export():
            pytest.skip(reason='aten.index.Tensor test is supported only on torch.export()')
        self._test(*self.create_model(indices_list, safe), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'input_shape': input_shape})

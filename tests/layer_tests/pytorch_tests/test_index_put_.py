# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndexPut_SingleIndices(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, indices, accumulate):

        class aten_index_put_(torch.nn.Module):

            def __init__(self, indices, accumulate):
                super().__init__()
                self.indices = indices
                self.accumulate = accumulate

            def forward(self, input_tensor, values):
                input_tensor.index_put_((self.indices,), values, self.accumulate)
                return input_tensor

        ref_net = None

        return aten_index_put_(indices, accumulate), ref_net, "aten::index_put_"

    @pytest.mark.parametrize('input_data', ({'input_tensor': np.random.randn(5).astype(np.float32),
                                             'values': np.array(11).astype(np.float32)},
                                             {'input_tensor': np.random.randn(3, 3).astype(np.float32),
                                             'values': np.array([10, 11, 12]).astype(np.float32)}))
    @pytest.mark.parametrize('indices', (torch.tensor([0], dtype=torch.long),
                                         torch.tensor([-1, -2], dtype=torch.long),
                                         torch.tensor([0, -1, -2], dtype=torch.long),
                                         torch.tensor([1, 2], dtype=torch.long),
                                         torch.tensor([0, 1, 2], dtype=torch.long)))
    @pytest.mark.parametrize('accumulate', (True, False))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_single_indices(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = input_data['input_tensor']
        self.values = input_data['values']
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version)

class TestIndexPut_ManyIndices(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor, self.values)

    def create_model(self, indices, accumulate):

        class aten_index_put_(torch.nn.Module):

            def __init__(self, indices, accumulate):
                super().__init__()
                self.indices_first = indices[0]
                self.indices_second = indices[1]
                self.accumulate = accumulate

            def forward(self, input_tensor, values):
                input_tensor.index_put_((self.indices_first, self.indices_second), values, self.accumulate)
                return input_tensor

        ref_net = None

        return aten_index_put_(indices, accumulate), ref_net, "aten::index_put_"

    @pytest.mark.parametrize('input_data', ({'input_tensor': np.random.randn(3, 3).astype(np.float32),
                                             'values': np.array(12).astype(np.float32)},
                                             {'input_tensor': np.random.randn(3, 3, 3).astype(np.float32),
                                             'values': np.array([10, 11, 12]).astype(np.float32)},))
    @pytest.mark.parametrize('indices', ((torch.tensor([0], dtype=torch.long),
                                          torch.tensor([2], dtype=torch.long)),
                                         (torch.tensor([1, 2], dtype=torch.long),
                                          torch.tensor([0, 1], dtype=torch.long)),
                                         (torch.tensor([0, 1], dtype=torch.long),
                                          torch.tensor([0, 1], dtype=torch.long)),
                                          (torch.tensor([0], dtype=torch.long),
                                          torch.tensor([-2], dtype=torch.long)),
                                         (torch.tensor([-1, -2], dtype=torch.long),
                                          torch.tensor([0, 1], dtype=torch.long)),
                                         (torch.tensor([0, -1], dtype=torch.long),
                                          torch.tensor([0, -1], dtype=torch.long))))
    @pytest.mark.parametrize('accumulate', (True, False))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_index_put_many_indices(self, ie_device, precision, ir_version, input_data, indices, accumulate):
        self.input_tensor = input_data['input_tensor']
        self.values = input_data['values']
        self._test(*self.create_model(indices, accumulate), ie_device, precision, ir_version)
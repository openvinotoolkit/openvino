# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytorch_layer_test_class import PytorchLayerTest
import numpy as np
import random
import torch


@pytest.mark.parametrize('input_data', ({'repeats': 1, 'dim': 0},
                                        {'repeats': 2, 'dim': 2},
                                        {'repeats': [2, 3], 'dim': 1},
                                        {'repeats': [3, 2, 1], 'dim': 3},
                                        {'repeats': 2, 'dim': None},
                                        {'repeats': [36], 'dim': None}))
class TestRepeatInterleaveConstRepeats(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 2, 3, 3),)

    def create_model_const_repeat(self, repeats, dim):
        class aten_repeat_interleave_const_repeat(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.repeats = torch.tensor(repeats, dtype=torch.int)
                self.dim = dim

            def forward(self, input_tensor):
                return input_tensor.repeat_interleave(self.repeats, self.dim)

        ref_net = None

        return aten_repeat_interleave_const_repeat(), ref_net, "aten::repeat_interleave"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_repeat_interleave_const_repeats(self, ie_device, precision, ir_version, input_data):
        repeats = input_data['repeats']
        if type(repeats) is list and len(repeats) == 1:
            repeats = [random.randint(1, 5) for _ in range(repeats[0])]

        dim = input_data['dim']
        self._test(*self.create_model_const_repeat(repeats, dim),
                   ie_device, precision, ir_version)

@pytest.mark.parametrize('input_data', ({'repeats': np.array([1]).astype(np.int32), 'dim': 0},
                                        {'repeats': np.array(1).astype(np.int32), 'dim': 1},
                                        {'repeats': np.array([2]).astype(np.int32), 'dim': 2},
                                        {'repeats': np.array(2).astype(np.int32), 'dim': 1},
                                        {'repeats': np.array([3]).astype(np.int32), 'dim': None}))
class TestRepeatInterleaveNonConstRepeats(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 2, 3, 3), self.repeats)

    def create_model_non_const_repeat(self, dim):
        class aten_repeat_interleave_non_const_repeat(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.dim = dim

            def forward(self, input_tensor, repeats):
                return input_tensor.repeat_interleave(repeats, self.dim)

        ref_net = None

        return aten_repeat_interleave_non_const_repeat(), ref_net, "aten::repeat_interleave"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_repeat_interleave_non_const_repeats(self, ie_device, precision, ir_version, input_data):
        self.repeats = input_data['repeats']
        dim = input_data['dim']
        self._test(*self.create_model_non_const_repeat(dim),
                   ie_device, precision, ir_version, dynamic_shapes=False)


class TestRepeatInterleaveTensorRepeats(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(10), self.repeats)

    def create_model_non_const_repeat(self, dim):
        class aten_repeat_interleave_non_const_repeat(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.dim = dim

            def forward(self, input_tensor, repeats):
                return input_tensor.repeat_interleave(repeats, self.dim)

        return aten_repeat_interleave_non_const_repeat(), None, "aten::repeat_interleave"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('input_data', ({'repeats': np.array([2, 3, 4, 1, 2, 4, 6, 7, 4, 3]).astype(np.int32), 'dim': None},
                                            {'repeats': np.array([3, 7, 1, 9, 4, 6, 2, 8, 5, 0]).astype(np.int32), 'dim': None}))
    def test_repeat_interleave_tensor_repeats(self, ie_device, precision, ir_version, input_data):
        self.repeats = input_data['repeats']
        dim = input_data['dim']
        self._test(*self.create_model_non_const_repeat(dim),
                   ie_device, precision, ir_version, dynamic_shapes=False)

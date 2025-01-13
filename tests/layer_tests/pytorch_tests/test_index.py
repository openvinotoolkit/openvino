# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestIndex(PytorchLayerTest):
    def _prepare_input(self, input_shape, idx):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32), idx)

    def create_model(self, model="list"):
        import torch

        class aten_index_list(torch.nn.Module):

            def forward(self, x, idx):
                return x[idx]

        class aten_index_getitem(torch.nn.Module):

            def forward(self, x, idx):
                return x.__getitem__(idx)

        class aten_index_list_bool(torch.nn.Module):

            def forward(self, x, idx):
                return x[idx.to(torch.bool)]

        class aten_index_getitem_bool(torch.nn.Module):

            def forward(self, x, idx):
                return x.__getitem__(idx.to(torch.bool))
        cases = {
            "list": aten_index_list,
            "getitem": aten_index_getitem,
            "list_with_bool": aten_index_list_bool,
            "getitem_with_bool": aten_index_getitem_bool
        }

        aten_index = cases[model]

        ref_net = None

        return aten_index(), ref_net, "aten::index"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("case", ["list", "getitem"])
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ((1,), np.array(0).astype(int)),
        ([2, 3], np.array(-1).astype(int)),
        ([4, 5, 6], np.array((1, 2)).astype(int)),
        ([7, 8, 9], np.array((-1,  2, -3)).astype(int)),
        ([2, 2, 3, 4], np.array((1,)).astype(int))])
    def test_index(self, input_shape, idx, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "idx": idx})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("case", ["getitem_with_bool", "list_with_bool"])
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ((1, 2), np.array([[1, 0]]).astype(bool)),
        ((2, 2, 5), np.zeros([2, 2, 5]).astype(bool)),
        ((2, 2, 5), np.ones([2, 2, 5]).astype(bool)),
        ((2, 2, 5), np.random.rand(2, 2, 5) > 0)
    ])
    def test_index_bool(self, input_shape, idx, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape, "idx": idx})


class TestIndexRange(PytorchLayerTest):
    def _prepare_input(self, input_shape, idx):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32), np.array(idx).astype(np.int32))

    def create_model(self):
        import torch

        class aten_index_arange(torch.nn.Module):

            def forward(self, x, y):
                x = x.reshape(x.shape[0], -1)
                return x[torch.arange(x.shape[0]), y]

        ref_net = None

        return aten_index_arange(), ref_net, "aten::index"

    def create_model2(self):
        import torch

        class aten_index_arange(torch.nn.Module):

            def forward(self, x, y):
                x = x.reshape(x.shape[0], x.shape[1], -1, 1)
                return x[torch.arange(x.shape[0]), y]

        ref_net = None

        return aten_index_arange(), ref_net, "aten::index"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "idx"), (
        ((1, 1), [0]),
        ([2, 3], [1, 2]),
        ([7, 8, 9], [1]),
        ([2, 2, 3, 4], [0])))
    def test_index_range(self, input_shape, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, kwargs_to_prepare_input={
                   "input_shape": input_shape, "idx": idx}, trace_model=True, dynamic_shapes=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "idx"), (
        ((1, 1), [0]),
        ([2, 3], [1, 2]),
        ([7, 8, 9], [1]),
        ([2, 2, 3, 4], [0])))
    def test_index_range_free_dims(self, input_shape, idx, ie_device, precision, ir_version):
        self._test(*self.create_model2(), ie_device, precision, ir_version, kwargs_to_prepare_input={
                   "input_shape": input_shape, "idx": idx}, trace_model=True, dynamic_shapes=False)


class TestIndexMask(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self):
        import torch

        class aten_index_mask(torch.nn.Module):
            def forward(self, x):
                return x[x > 0]

        ref_net = None

        return aten_index_mask(), ref_net, "aten::index"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape"), ((1, 1),
                                               [2, 3],
                                               [7, 8, 9],
                                               [2, 2, 3, 4]))
    def test_index_mask(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, kwargs_to_prepare_input={
                   "input_shape": input_shape}, trace_model=True, use_convert_model=True)


class TestIndexNone(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    class aten_index_list(torch.nn.Module):
        def __init__(self, idxs):
            super(TestIndexNone.aten_index_list, self).__init__()
            self.idxs = idxs

        def forward(self, x):
            return x[self.idxs]

    @pytest.mark.nightly
    @pytest.mark.parametrize(("input_shape,idxs"), [
        ((2, 3, 4, 5), (torch.unsqueeze(torch.randint(0, 2, [14], dtype=torch.int32), 1),)),
        ((2, 3, 4, 5), (torch.unsqueeze(torch.randint(0, 2, [14], dtype=torch.int32), 1), torch.randint(0, 3, [14], dtype=torch.int32))),
        ((2, 3, 4, 5), (None, None, torch.unsqueeze(torch.randint(0, 2, [14], dtype=torch.int32), 1), torch.randint(0, 3, [14], dtype=torch.int32))),
        ])
    def test_index(self, input_shape, idxs, ie_device, precision, ir_version):
        self._test(self.aten_index_list(idxs), None, "aten::index", ie_device, precision,
                   ir_version,kwargs_to_prepare_input={"input_shape": input_shape}, use_convert_model=True, trace_model=True)

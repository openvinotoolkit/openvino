# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

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
        cases = {
            "list": aten_index_list,
            "getitem": aten_index_getitem
        }

        aten_index = cases[model]

        ref_net = None

        return aten_index(), ref_net, "aten::index"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("case", ["list", "getitem"])
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ((1,), np.array(0).astype(np.int32)), 
        ([2, 3], np.array(-1).astype(np.int32)), 
        ([4, 5, 6], np.array((1, 2)).astype(np.int32)), 
        ([7, 8, 9], np.array((-1,  2, -3)).astype(np.int32)), 
        ([2, 2, 3, 4], np.array((1,)).astype(np.int32))])
    def test_index(self, input_shape, idx, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version, kwargs_to_prepare_input={"input_shape": input_shape, "idx": idx})
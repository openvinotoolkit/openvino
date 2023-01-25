# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestTupleConstruct(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (1, 10)).astype(np.float32),)

    def create_model(self, case):
        import torch

        class prim_tuple_construct_single_value(torch.nn.Module):

            def forward(self, x):
                return (x,)

        class prim_tuple_construct(torch.nn.Module):

            def forward(self, x):
                return (x, x + x)

        class prim_tuple_construct_with_none(torch.nn.Module):

            def forward(self, x):
                return (x, None, x + x, None)

        cases = {
            "single": prim_tuple_construct_single_value,
            "multiple": prim_tuple_construct,
            "none": prim_tuple_construct_with_none
        }

        ref_net = None
        model = cases[case]

        return model(), ref_net, "prim::TupleConstruct"

    @pytest.mark.parametrize("case", ["single", "multiple", "none"])
    @pytest.mark.nightly
    def test_tuple_construct(self, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version)

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

        class prim_tuple_construct_with_list(torch.nn.Module):

            def forward(self, x):
                return (x, [None, x + x], None)

        class prim_tuple_construct_with_list_and_tuple(torch.nn.Module):

            def forward(self, x):
                return (x, [None, x + x], None, (x + 1.0, x + 2.0, None))

        cases = {
            "single": prim_tuple_construct_single_value,
            "multiple": prim_tuple_construct,
            "none": prim_tuple_construct_with_none,
            "list": prim_tuple_construct_with_list,
            "list_and_tuple": prim_tuple_construct_with_list_and_tuple
        }

        ref_net = None
        model = cases[case]

        return model(), ref_net, "prim::TupleConstruct"

    @pytest.mark.parametrize("case", ["single", "multiple", "none", "list", "list_and_tuple"])
    @pytest.mark.nightly
    def test_tuple_construct(self, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version)
    

class TestTupleConstructTupleUnpack(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32),)

    def create_model(self):
        import torch

        class prim_tuple_construct_tuple_unpack(torch.nn.Module):

            def forward(self, x):
                x1, x2, x3, x4, x5 = self.prepare_input(x)
                return x1, x2, x3, x4, x5
            
            def prepare_input(self, x):
                return x, x + 2, None, x.reshape(-1), (x * 10).to(torch.int32)


        ref_net = None

        return prim_tuple_construct_tuple_unpack(), ref_net, ["prim::TupleConstruct", "prim::TupleUnpack"]

    @pytest.mark.nightly
    def test_tuple_construct_unpack(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, freeze_model=False)
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


class TestTupleUnpackParameterSingle(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        return ( (tensor_gen(), tensor_gen()), )

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):

            def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
                x1, x2 = x
                return x1, x2


        return model(), None, ["prim::TupleUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestTupleUnpackParameterNested(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        return ( ((tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen())), )

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):

            def forward(self, x: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
                x1, x2 = x
                y1, y2 = x1
                y3, y4 = x2
                return y1, y2, y3, y4


        return model(), None, ["prim::TupleUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestTupleUnpackParameterMultiple(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        return ( (tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen()) )

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):

            def forward(self, x: Tuple[torch.Tensor, torch.Tensor], y: Tuple[torch.Tensor, torch.Tensor]):
                z1, z2 = x
                z3, z4 = y
                return z1, z2, z3, z4


        return model(), None, ["prim::TupleUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)

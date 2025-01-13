# Copyright (C) 2018-2025 Intel Corporation
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

        class prim_tuple_construct_with_tensor_tail(torch.nn.Module):

            def forward(self, x):
                return ((x, x + x), x + x + x)

        class prim_tuple_construct_with_list_and_tuple(torch.nn.Module):

            def forward(self, x):
                return (x, [None, x + x], None, (x + 1.0, x + 2.0, None))

        cases = {
            "single": prim_tuple_construct_single_value,
            "multiple": prim_tuple_construct,
            "none": prim_tuple_construct_with_none,
            "list": prim_tuple_construct_with_list,
            "tensor_tail": prim_tuple_construct_with_tensor_tail,
            "list_and_tuple": prim_tuple_construct_with_list_and_tuple
        }

        ref_net = None
        model = cases[case]

        return model(), ref_net, "prim::TupleConstruct"

    @pytest.mark.parametrize("case", ["single", "multiple", "none", "list", "tensor_tail", "list_and_tuple"])
    @pytest.mark.nightly
    def test_tuple_construct(self, case, ie_device, precision, ir_version):
        self._test(*self.create_model(case), ie_device, precision, ir_version, use_convert_model=True)


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
        self._test(*self.create_model(), ie_device,
                   precision, ir_version, freeze_model=False, use_convert_model=True)


class TestTupleUnpackParameterSingle(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        return ((tensor_gen(), tensor_gen()), )

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


class TestTupleUnpackParameterSingleMixed(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        # generate tensor with a different shape for easier mismatch detection in case of mixed input order

        def tensor_gen_2():
            return np.random.uniform(0, 50, (2, 3)).astype(np.float32)
        return (tensor_gen_2(), (tensor_gen(), tensor_gen()), tensor_gen_2())

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):

            def forward(self, y1, x: Tuple[torch.Tensor, torch.Tensor], y2):
                x1, x2 = x
                return x1, x2, y1, y2

        return model(), None, ["prim::TupleUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestTupleUnpackParameterNested(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)
        return (((tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen())), )

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
        return ((tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen()))

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


class TestTupleIndex(PytorchLayerTest):
    def _prepare_input(self):
        return np.random.uniform(0, 50, (1, 2, 10)).astype(np.float32)

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):
            def forward(self, x):
                return self.some_func((x, x))

            def some_func(self, x: Tuple[torch.Tensor, torch.Tensor]):
                return x[1] * 2, x[0] * 3

        return model(), None, "prim::TupleIndex"

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=False, freeze_model=False, use_convert_model=True)


class TestTcOutsideTuInsideIfBody(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 2, 10).astype(np.float32), np.random.randn(1, 2, 10).astype(np.float32))

    def create_model(self):
        import torch
        from typing import Tuple

        class model(torch.nn.Module):
            def forward(self, x, y):
                return self.some_func((x, y))

            def some_func(self, x: Tuple[torch.Tensor, torch.Tensor]):
                if x[0].numel() > 10:
                    n, m = x
                    return n * m
                else:
                    n, m = x
                    return n - m

        return model(), None, ["prim::TupleConstruct", "prim::TupleUnpack", "prim::If"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=False, freeze_model=False, use_convert_model=True)

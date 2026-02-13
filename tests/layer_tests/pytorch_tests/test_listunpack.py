# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestListUnpack(PytorchLayerTest):
    def _prepare_input(self):
        return (
            self.random.randn(8, 3, 512, 512),
            self.random.randn(1, 3, 224, 224),
            self.random.randn(10, 1, 8, 8),
            self.random.randn(1, 1, 1, 1),
        )

    def create_model_size_listunpack(self):
        class prim_listunpack(torch.nn.Module):
            def forward(self, in1, in2, in3, in4):
                a, b, c, d = in1.size()
                return a, b, c, d


        return (
            prim_listunpack(), "prim::ListUnpack",
        )

    def create_model_size_slice_listunpack(self, slices):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, slices):
                self.start = slices[0]
                self.stop = slices[1]
                self.step = slices[2]
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                a, b = in1.size()[self.start: self.stop: self.step]
                return a, b


        return prim_listunpack(slices), "prim::ListUnpack"

    def create_model_listconstruct_append_listunpack(self):
        class prim_listunpack(torch.nn.Module):
            def forward(self, in1, in2, in3, in4):
                in_list = [in1, in2]
                in_list.append(in3)
                in_list.append(in4)
                a, b, c, d = in_list
                return a, b, c, d


        return prim_listunpack(), "prim::ListUnpack"

    def create_model_listconstruct_getitem_listunpack(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                items: list[list[int]] = []
                items.append(in1.size())
                items.append(in2.size())
                items.append(in3.size())
                items.append(in4.size())
                getitem_0 = items[self.idx]
                a, b, c, d = getitem_0
                return a, b, c, d


        return prim_listunpack(idx), "prim::ListUnpack"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_size_listunpack(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model_size_listunpack(), ie_device, precision, ir_version
        )

    @pytest.mark.parametrize(
        "slices", [(0, 2, 1), (0, 4, 2), (-1, -3, -1), (-3, -1, 1)]
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_size_slice_listunpack(self, slices, ie_device, precision, ir_version):
        self._test(
            *self.create_model_size_slice_listunpack(slices),
            ie_device,
            precision,
            ir_version
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_listconstruct_append_listunpack(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model_listconstruct_append_listunpack(),
            ie_device,
            precision,
            ir_version
        )

    @pytest.mark.parametrize("idx", [-4, -3, -2, -1, 0, 1, 2, 3])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_listconstruct_getitem_listunpack(
            self, idx, ie_device, precision, ir_version
    ):
        self._test(
            *self.create_model_listconstruct_getitem_listunpack(idx),
            ie_device,
            precision,
            ir_version,
            use_convert_model=True,
        )


class TestMeshgridListUnpack(PytorchLayerTest):
    def _prepare_input(self):
        return (
            self.random.randn(3),
            self.random.randn(5),
            self.random.randn(7),
            self.random.randn(11),
        )

    def create_model_meshgrid_listunpack_1_in(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                (
                    a,
                ) = torch.meshgrid(in1, indexing=self.idx)
                return a


        return prim_listunpack(idx), ["aten::meshgrid", "prim::ListUnpack"]

    def create_model_meshgrid_listunpack_2_in(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                (
                    a,
                    b,
                ) = torch.meshgrid(in1, in2, indexing=self.idx)
                return a, b


        return prim_listunpack(idx), ["aten::meshgrid", "prim::ListUnpack"]

    def create_model_meshgrid_listunpack_3_in(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                a, b, c = torch.meshgrid(in1, in2, in3, indexing=self.idx)
                return a, b, c


        return prim_listunpack(idx), ["aten::meshgrid", "prim::ListUnpack"]

    def create_model_meshgrid_listunpack_4_in(self, idx):
        class prim_listunpack(torch.nn.Module):
            def __init__(self, idx):
                self.idx = idx
                super().__init__()

            def forward(self, in1, in2, in3, in4):
                a, b, c, d = torch.meshgrid(
                    in1, in2, in3, in4, indexing=self.idx)
                return a, b, c, d


        return prim_listunpack(idx), ["aten::meshgrid", "prim::ListUnpack"]

    @pytest.mark.parametrize("idx", ["ij", "xy"])
    @pytest.mark.parametrize("inp", [1, 2, 3, 4])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_meshgrid_listunpack(self, idx, inp, ie_device, precision, ir_version):
        func = getattr(self, f"create_model_meshgrid_listunpack_{inp}_in")
        self._test(*func(idx), ie_device, precision, ir_version)


class TestMeshgridListUnpackStack(PytorchLayerTest):
    def _prepare_input(self):
        return (
            self.random.randn(28, 28),
        )

    def create_model(self):
        class meshgrid_model(torch.nn.Module):
            def forward(self, x):
                h, w = x.shape
                coords1, coords2 = torch.meshgrid(
                    torch.arange(h), torch.arange(w), indexing="ij")
                coords = torch.stack([coords2, coords1], dim=0)
                return coords.float()

        return meshgrid_model(), ["aten::meshgrid", "aten::stack", "prim::ListUnpack"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_meshgrid_subgraph(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestListUnpackParameterSingle(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return self.random.uniform(0, 50, (1, 2, 10), dtype=np.float32)
        return ((tensor_gen(), tensor_gen()), )

    def create_model(self):
        import torch

        class model(torch.nn.Module):

            def forward(self, x: list[torch.Tensor]):
                x1, x2 = x
                return x1, x2

        return model(), ["prim::ListUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestListUnpackParameterSingleMixed(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return self.random.uniform(0, 50, (1, 2, 10), dtype=np.float32)
        # generate tensor with a different shape for easier mismatch detection in case of mixed input order

        def tensor_gen_2():
            return self.random.uniform(0, 50, (2, 3), dtype=np.float32)
        return (tensor_gen_2(), (tensor_gen(), tensor_gen()), tensor_gen_2())

    def create_model(self):
        import torch

        class model(torch.nn.Module):

            def forward(self, y1, x: list[torch.Tensor], y2):
                x1, x2 = x
                return x1, x2, y1, y2

        return model(), ["prim::ListUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestListUnpackParameterNested(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return self.random.uniform(0, 50, (1, 2, 10), dtype=np.float32)
        return (((tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen())), )

    def create_model(self):
        import torch

        class model(torch.nn.Module):

            def forward(self, x: list[list[torch.Tensor]]):
                x1, x2 = x
                y1, y2 = x1
                y3, y4 = x2
                return y1, y2, y3, y4

        return model(), ["prim::ListUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestListUnpackParameterMultiple(PytorchLayerTest):
    def _prepare_input(self):
        def tensor_gen():
            return self.random.uniform(0, 50, (1, 2, 10), dtype=np.float32)
        return ((tensor_gen(), tensor_gen()), (tensor_gen(), tensor_gen()))

    def create_model(self):
        import torch

        class model(torch.nn.Module):

            def forward(self, x: list[torch.Tensor], y: list[torch.Tensor]):
                z1, z2 = x
                z3, z4 = y
                return z1, z2, z3, z4

        return model(), ["prim::ListUnpack"]

    @pytest.mark.nightly
    def test(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestComplexListUnpack(PytorchLayerTest):
    """Test prim::ListUnpack with complex tensors (ComplexTypeMark propagation).

    This tests the changes in list_unpack.cpp that handle ComplexTypeMark
    propagation through ListUnpack operations.
    """

    def _prepare_input(self):
        return (self.random.randn(2, 4, 2),)

    def create_model(self):
        class ComplexListUnpack(torch.nn.Module):
            def __init__(self, rng):
                super().__init__()
                freqs = rng.torch_randn(4, 2)
                complex_freqs = torch.view_as_complex(freqs)
                self.register_buffer('freqs', torch.view_as_real(complex_freqs))

            def forward(self, x):
                cx = torch.view_as_complex(x)
                cf = torch.view_as_complex(self.freqs)
                result = cx * cf
                return torch.view_as_real(result)

        return ComplexListUnpack(self.random), "aten::mul"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_complex_list_unpack(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   trace_model=True)

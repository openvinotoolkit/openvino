# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestEinsumBatchMatMul(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (np.random.randn(5, 2, 3).astype(np.float32), np.random.randn(5, 3, 4).astype(np.float32),)

    def create_model(self):
        import torch

        class EinsumModelBatchMatmul(torch.nn.Module):
            def forward(self, x, y):
                eqn = "bij, bjk -> bik"
                return torch.einsum(eqn, x, y)

        ref_net = None

        return EinsumModelBatchMatmul(), ref_net, "aten::einsum"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_einsum_batch_matmul(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestEinsumBatchDiagonal(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (np.random.randn(3, 5, 5).astype(np.float32),)

    def create_model(self):
        import torch

        class EinsumModelBatchDiagonal(torch.nn.Module):
            def forward(self, x):
                eqn = "kii -> ki"
                return torch.einsum(eqn, x)

        ref_net = None

        return EinsumModelBatchDiagonal(), ref_net, "aten::einsum"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason='OpenVINO CPU plugin does not support einsum diagonal')
    def test_einsum_batch_diagonal(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, dynamic_shapes=False)


class TestEinsumInnerProd(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (np.random.randn(5).astype(np.float32), np.random.randn(5).astype(np.float32))

    def create_model(self):
        import torch

        class EinsumModelInnerProd(torch.nn.Module):
            def forward(self, x, y):
                eqn = "i,i"
                return torch.einsum(eqn, x, y)

        ref_net = None

        return EinsumModelInnerProd(), ref_net, "aten::einsum"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_einsum_inner_prod(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestEinsumTranspose(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np

        return (np.random.randn(3, 5).astype(np.float32),)

    def create_model(self):
        import torch

        class EinsumModelTranspose(torch.nn.Module):
            def forward(self, x):
                eqn = "ij->ji"
                return torch.einsum(eqn, x)

        ref_net = None

        return EinsumModelTranspose(), ref_net, "aten::einsum"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_einsum_transpose(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)

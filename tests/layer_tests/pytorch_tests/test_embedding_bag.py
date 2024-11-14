# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestEmbeddingBag1dOffsets(PytorchLayerTest):
    def _prepare_input(self, indicies_dtype, per_sample_weights=False):
        import numpy as np

        indices = np.array([2, 2, 2, 2, 4, 3, 2, 9]).astype(indicies_dtype)
        weights = np.random.randn(10, 10).astype(np.float32)
        offsets = np.array([0, 4, 4]).astype(indicies_dtype)
        if per_sample_weights:
            per_sample_weights = np.random.randn(*indices.shape).astype(np.float32)
            return (indices, weights, offsets, per_sample_weights)
        return (indices, weights, offsets)

    def create_model(self, mode, per_sample_weights):
        import torch
        import torch.nn.functional as F

        class aten_embedding_bag(torch.nn.Module):
            def __init__(self, mode=None, per_sample_weights=False) -> None:
                super().__init__()
                self.mode = mode
                if per_sample_weights:
                    self.forward = self.forward_offsets_per_sample_weights

            def forward(self, indicies, weight, offsets):
                return F.embedding_bag(indicies, weight, offsets, mode=self.mode)

            def forward_offsets_per_sample_weights(
                self, indicies, weight, offsets, per_sample_wights
            ):
                return F.embedding_bag(
                    indicies,
                    weight,
                    offsets,
                    mode=self.mode,
                    per_sample_weights=per_sample_wights,
                )

        ref_net = None

        return (
            aten_embedding_bag(mode, per_sample_weights),
            ref_net,
            "aten::embedding_bag",
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    @pytest.mark.parametrize(
        "mode, per_sample_weights", [("mean", False), ("sum", False), ("sum", True)]
    )
    @pytest.mark.xfail(
        condition=platform.system() == "Darwin" and platform.machine() == "arm64",
        reason="Ticket - 122715",
    )
    def test_embedding_bag(
        self, ie_device, precision, ir_version, indicies_dtype, mode, per_sample_weights
    ):
        self._test(
            *self.create_model(mode, per_sample_weights),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "indicies_dtype": indicies_dtype,
                "per_sample_weights": per_sample_weights,
            },
            trace_model=True,
            dynamic_shapes=not per_sample_weights and ie_device != "GPU"
        )


class TestEmbeddingBag2d(PytorchLayerTest):
    def _prepare_input(self, indicies_size, indicies_dtype, per_sample_weights):
        import numpy as np

        indices = np.random.randint(0, 9, size=indicies_size).astype(indicies_dtype)
        weights = np.random.randn(10, 10).astype(np.float32)
        if per_sample_weights:
            per_sample_weights = np.random.randn(*indices.shape).astype(np.float32)
            return (indices, weights, per_sample_weights)
        return (indices, weights)

    def create_model(self, mode, per_sample_weights):
        import torch
        import torch.nn.functional as F

        class aten_embedding_bag(torch.nn.Module):
            def __init__(self, mode=None, per_sample_weights=False) -> None:
                super().__init__()
                self.mode = mode
                if per_sample_weights:
                    self.forward = self.forward_per_sample_weights

            def forward(self, indicies, weight):
                return F.embedding_bag(indicies, weight, mode=self.mode)

            def forward_per_sample_weights(self, indicies, weight, per_sample_wights):
                return F.embedding_bag(
                    indicies,
                    weight,
                    mode=self.mode,
                    per_sample_weights=per_sample_wights,
                )

        ref_net = None

        return (
            aten_embedding_bag(mode, per_sample_weights),
            ref_net,
            "aten::embedding_bag",
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("indicies_size", [[1, 1], [2, 5], [3, 10], [4, 7]])
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    @pytest.mark.parametrize(
        "mode, per_sample_weights", [("mean", False), ("sum", False), ("sum", True)]
    )
    @pytest.mark.xfail(
        condition=platform.system() == "Darwin" and platform.machine() == "arm64",
        reason="Ticket - 122715",
    )
    def test_embedding_bag(
        self,
        ie_device,
        precision,
        ir_version,
        indicies_dtype,
        indicies_size,
        mode,
        per_sample_weights,
    ):
        self._test(
            *self.create_model(mode, per_sample_weights),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "indicies_size": indicies_size,
                "indicies_dtype": indicies_dtype,
                "per_sample_weights": per_sample_weights,
            },
            trace_model=True,
            dynamic_shapes=not per_sample_weights and ie_device != "GPU"
        )


class TestEmbeddingBagPretrained(PytorchLayerTest):
    def _prepare_input(self, indicies_size, indicies_dtype, per_sample_weights):
        import numpy as np

        indices = np.random.randint(0, 9, size=indicies_size).astype(indicies_dtype)
        if per_sample_weights:
            per_sample_weights = np.random.randn(*indices.shape).astype(np.float32)
            return (indices, per_sample_weights)
        return (indices,)

    def create_model(self, mode, per_sample_weights):
        import torch
        import numpy as np

        class aten_embedding_bag(torch.nn.Module):
            def __init__(self, mode=None, per_sample_weights=False) -> None:
                super().__init__()
                self.mode = mode
                weights = torch.from_numpy(np.random.randn(10, 10).astype(np.float32))

                self.embeddings = torch.nn.EmbeddingBag.from_pretrained(
                    weights, mode=mode
                )
                if per_sample_weights:
                    self.forward = self.forward_per_sample_weights

            def forward(self, indicies):
                return self.embeddings(indicies)

            def forward_per_sample_weights(self, indicies, per_sample_wights):
                return (
                    self.embeddings(indicies, per_sample_weights=per_sample_wights),
                )

        ref_net = None

        return (
            aten_embedding_bag(mode, per_sample_weights),
            ref_net,
            "aten::embedding_bag",
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("indicies_size", [[1, 1], [2, 5], [3, 10], [4, 7]])
    @pytest.mark.parametrize("indicies_dtype", ["int", "int32"])
    @pytest.mark.parametrize(
        "mode, per_sample_weights", [("mean", False), ("sum", False), ("sum", True)]
    )
    @pytest.mark.xfail(
        condition=platform.system() == "Darwin" and platform.machine() == "arm64",
        reason="Ticket - 122715",
    )
    def test_embedding_bag(
        self,
        ie_device,
        precision,
        ir_version,
        indicies_dtype,
        indicies_size,
        mode,
        per_sample_weights,
    ):
        self._test(
            *self.create_model(mode, per_sample_weights),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "indicies_size": indicies_size,
                "indicies_dtype": indicies_dtype,
                "per_sample_weights": per_sample_weights,
            },
            trace_model=True,
            dynamic_shapes=not per_sample_weights and ie_device != "GPU"
        )

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0i

import pytest

from pytorch_layer_test_class import PytorchLayerTest

class TestExpm1(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype=None):
        import numpy as np
        return [np.array(inputs).astype(dtype)]

    def create_model(self, dtype=None):
        import torch    
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
        }

        class aten_expm1(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.expm1(x.type(self.dtype))

        dtype = dtype_map.get(dtype)
        model_class = aten_expm1(dtype)

        ref_net = None

        return model_class, ref_net, "aten::expm1"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("inputs", [[0, 1, 2, 3, 4, 5], [-2, -1, 0, 1, 2, 3], [1, 2, 3, 4, 5, 6]])
    def test_expm1(self, dtype, inputs, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype}
        )
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTypeAs(PytorchLayerTest):
    def _prepare_input(self, input_dtype=np.float32, cast_dtype=np.float32):
        input_data = np.random.randint(127, size=(1, 3, 224, 224))
        return (input_data.astype(input_dtype), input_data.astype(cast_dtype))

    def create_model(self):
        import torch

        class aten_type_as(torch.nn.Module):

            def forward(self, x, y):
                return x.type_as(y)

        ref_net = None

        return aten_type_as(), ref_net, "aten::type_as"

    @pytest.mark.parametrize("input_dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint8])
    @pytest.mark.parametrize("cast_dtype", [np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint8])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_type_as(self, input_dtype, cast_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_dtype": input_dtype, "cast_dtype": cast_dtype})

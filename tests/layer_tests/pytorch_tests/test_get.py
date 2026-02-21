# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestGet(PytorchLayerTest):
    def _prepare_input(self, input_shape):
        import numpy as np
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self, key, case="dict_get"):
        class aten_dict_get(torch.nn.Module):
            def __init__(self, key):
                super().__init__()
                self.key = key

            def forward(self, x):
                data = {0: x, 1: x + 1}
                return data.get(self.key)

        class aten_dict_get_with_default(torch.nn.Module):
            def __init__(self, key):
                super().__init__()
                self.key = key

            def forward(self, x):
                data = {0: x, 1: x + 1}
                return data.get(self.key, x * 2)

        class aten_dict_get_str(torch.nn.Module):
            def __init__(self, key):
                super().__init__()
                self.key = key

            def forward(self, x):
                data = {"a": x, "b": x + 1}
                return data.get(self.key)

        class aten_dict_get_str_with_default(torch.nn.Module):
            def __init__(self, key):
                super().__init__()
                self.key = key

            def forward(self, x):
                data = {"a": x, "b": x + 1}
                return data.get(self.key, x * 2)

        ref_net = None
        op_cls = {
            "dict_get": (aten_dict_get, ["aten::get"]),
            "dict_get_with_default": (aten_dict_get_with_default, ["aten::get"]),
            "dict_get_str": (aten_dict_get_str, ["aten::get"]),
            "dict_get_str_with_default": (aten_dict_get_str_with_default, ["aten::get"]),
        }
        op, op_in_graph = op_cls[case]

        return op(key), ref_net, op_in_graph

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ([2, 3], 0),
        ([2, 3], 1),
        ([3, 4, 5], 0),
        ([3, 4, 5], 1),
    ])
    def test_get(self, input_shape, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(idx, "dict_get"), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "idx"), [
        ([2, 3], 2),
        ([3, 4, 5], 2),
    ])
    def test_get_with_default(self, input_shape, idx, ie_device, precision, ir_version):
        self._test(*self.create_model(idx, "dict_get_with_default"), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "key"), [
        ([2, 3], "a"),
        ([3, 4, 5], "b"),
    ])
    def test_get_str(self, input_shape, key, ie_device, precision, ir_version):
        self._test(*self.create_model(key, "dict_get_str"), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("input_shape", "key"), [
        ([2, 3], "c"),
        ([3, 4, 5], "d"),
    ])
    def test_get_str_with_default(self, input_shape, key, ie_device, precision, ir_version):
        self._test(*self.create_model(key, "dict_get_str_with_default"), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"input_shape": input_shape})

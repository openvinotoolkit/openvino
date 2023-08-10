# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest



class aten_cat(torch.nn.Module):
    def __init__(self, out) -> None:
        super().__init__()
        if out:
            self.forward = self.forward_out

    def forward(self, x):
        return torch.cat(self.prepare_input(x), 1)

    def forward_out(self, x, out):
        return torch.cat(self.prepare_input(x), 1, out=out), out

    def prepare_input(self, x):
        return [x, x]


class aten_append_cat(aten_cat):
    def prepare_input(self, x):
        list = []
        list.append(x)
        list.append(x)
        return list


class aten_loop_append_cat(aten_cat):
    def prepare_input(self, x):
        list = []
        for i in range(3):
            list.append(x)
        return list


class aten_add_cat(aten_cat):
    def prepare_input(self, x):
        list = [x, x]
        list2 = list + [x, x]
        return list2


class TestCat(PytorchLayerTest):
    def _prepare_input(self, out=False, num_repeats=2):
        import numpy as np
        data = np.random.randn(2, 1, 3)
        if not out:
            return (data, )
        concat = [data for _ in range(num_repeats)]
        out = np.zeros_like(np.concatenate(concat, axis=1))
        return (data, out)
        

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_cat(self, out, ie_device, precision, ir_version):
        self._test(aten_cat(out), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "num_repeats": 2})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_append_cat(self, out, ie_device, precision, ir_version):
        self._test(aten_append_cat(out), None, ["aten::cat", "aten::append", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "num_repeats": 2})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="Transformation RemoveMultiSubGraphOpDanglingParamsResults doesn't support removing unused merged inputs, ticket 112833.")
    @pytest.mark.parametrize("out", [False, True])
    def test_loop_append_cat(self, out, ie_device, precision, ir_version):
        self._test(aten_loop_append_cat(out), None, ["aten::cat", "aten::append", "prim::ListConstruct", "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False,  kwargs_to_prepare_input={"out": out, "num_repeats": 3})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_add_cat(self, out, ie_device, precision, ir_version):
        self._test(aten_add_cat(out), None, ["aten::cat", "aten::add", "prim::ListConstruct"],
                   ie_device, precision, ir_version, freeze_model=False,  kwargs_to_prepare_input={"out": out, "num_repeats": 4})

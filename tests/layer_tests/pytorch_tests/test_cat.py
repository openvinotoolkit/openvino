# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest



class aten_cat(torch.nn.Module):
    def forward(self, x):
        return torch.cat(self.prepare_input(x), 1)

    def prepare_input(self, x):
        return [x, x]

class aten_cat_out(aten_cat):
    def forward(self, x, out):
        return torch.cat(self.prepare_input(x), 1, out=out), out

class aten_append_cat(aten_cat):
    def prepare_input(self, x):
        list = []
        list.append(x)
        list.append(x)
        return list

class aten_append_cat_out(aten_cat_out):
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


class aten_loop_append_cat_out(aten_cat_out):
    def prepare_input(self, x):
        list = []
        for i in range(3):
            list.append(x)
        return list

class aten_add_cat(aten_cat):
    def forward(self, x):
        list1 = self.prepare_input(x)
        list2 = self.prepare_input(x)
        return torch.cat(list1 + list2, dim=1)


class aten_add_cat_out(aten_cat_out):
    def forward(self, x, out):
        list1 = self.prepare_input(x)
        list2 = self.prepare_input(x)
        return torch.cat(list1 + list2, dim=1, out=out)

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
        model = aten_cat() if not out else aten_cat_out()
        self._test(model, None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "num_repeats": 2})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_append_cat(self, out, ie_device, precision, ir_version):
        model = aten_append_cat() if not out else aten_append_cat_out()
        self._test(model, None, ["aten::cat", "aten::append", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "num_repeats": 2})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="Transformation RemoveMultiSubGraphOpDanglingParamsResults doesn't support removing unused merged inputs, ticket 112833.")
    @pytest.mark.parametrize("out", [False, True])
    def test_loop_append_cat(self, out, ie_device, precision, ir_version):
        model = aten_loop_append_cat() if not out else aten_loop_append_cat_out()
        self._test(model, None, ["aten::cat", "aten::append", "prim::ListConstruct", "prim::Loop"],
                   ie_device, precision, ir_version, freeze_model=False,  kwargs_to_prepare_input={"out": out, "num_repeats": 3})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_add_cat(self, out, ie_device, precision, ir_version):
        model = aten_add_cat() if not out else aten_add_cat_out()
        self._test(model, None, ["aten::cat", "aten::add", "prim::ListConstruct"],
                   ie_device, precision, ir_version, freeze_model=False,  kwargs_to_prepare_input={"out": out, "num_repeats": 4})

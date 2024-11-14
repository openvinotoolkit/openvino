# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

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


class TestCatAlignTypes(PytorchLayerTest):
    def _prepare_input(self, in_types):
        in_vals = []
        for i in range(len(in_types)):
            dtype = in_types[i]
            in_vals.append(np.random.randn(2, 1, 3).astype(dtype))
        return in_vals

    def create_model(self, in_count):
        class aten_align_types_cat_two_args(torch.nn.Module):
            def forward(self, x, y):
                ins = [x, y]
                return torch.cat(ins, 1)

        class aten_align_types_cat_three_args(torch.nn.Module):
            def forward(self, x, y, z):
                ins = [x, y, z]
                return torch.cat(ins, 1)

        if in_count == 2:
            return aten_align_types_cat_two_args()

        if in_count == 3:
            return aten_align_types_cat_three_args()

    @pytest.mark.parametrize(("in_types"), [
        # Two inputs
        (np.float32, np.int32),
        (np.int32, np.float32),
        (np.float16, np.float32),
        (np.int16, np.float16),
        (np.int32, np.int64),
        # # Three inputs
        (np.float32, np.int32, np.int32),
        (np.float32, np.int32, np.float32),
        (np.int32, np.float32, np.int32),
        (np.float32, np.int32, np.int16),
        (np.int32, np.float32, np.int16),
        (np.int16, np.int32, np.int16),
        (np.float16, np.float32, np.float16),
        (np.float32, np.float16, np.float32),
        (np.float16, np.int32, np.int16),
        (np.int16, np.float16, np.int16),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_cat(self, ie_device, precision, ir_version, in_types, trace_model):
        self._test(self.create_model(len(in_types)), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model)


class TestCatAlignTypesPT(PytorchLayerTest):
    def _prepare_input(self, in_types):
        in_vals = [np.random.randn(2, 2, 3).astype(in_types[0])]
        return in_vals

    def create_model_param_first(self, in_types):
        class aten_align_types_cat_two_args(torch.nn.Module):
            def __init__(self):
                super(aten_align_types_cat_two_args, self).__init__()
                self.y = torch.randn(2, 1, 3).to(in_types[1])

            def forward(self, x):
                x_ = torch.split(x, 1, 1)[1]
                ins = [x_, self.y]
                return torch.cat(ins, 1)

        class aten_align_types_cat_three_args(torch.nn.Module):
            def __init__(self):
                super(aten_align_types_cat_three_args, self).__init__()
                self.y = torch.randn(2, 1, 3).to(in_types[1])
                self.z = torch.randn(2, 1, 3).to(in_types[2])

            def forward(self, x):
                x_ = torch.split(x, 1, 1)[1]
                ins = [x_, self.y, self.z]
                return torch.cat(ins, 1)

        in_count = len(in_types)
        if in_count == 2:
            return aten_align_types_cat_two_args()

        if in_count == 3:
            return aten_align_types_cat_three_args()

    def create_model_param_mid(self, in_types):
        class aten_align_types_cat_three_args(torch.nn.Module):
            def __init__(self):
                super(aten_align_types_cat_three_args, self).__init__()
                self.x = torch.randn(2, 1, 3).to(in_types[1])
                self.z = torch.randn(2, 1, 3).to(in_types[2])

            def forward(self, y):
                y_ = torch.split(y, 1, 1)[1]
                ins = [self.x, y_, self.z]
                return torch.cat(ins, 1)
        return aten_align_types_cat_three_args()

    def create_model_param_last(self, in_types):
        class aten_align_types_cat_three_args(torch.nn.Module):
            def __init__(self):
                super(aten_align_types_cat_three_args, self).__init__()
                self.x = torch.randn(2, 1, 3).to(in_types[1])
                self.y = torch.randn(2, 1, 3).to(in_types[2])

            def forward(self, z):
                z_ = torch.split(z, 1, 1)[1]
                ins = [self.x, self.y, z_]
                return torch.cat(ins, 1)
        return aten_align_types_cat_three_args()

    @pytest.mark.parametrize(("in_types"), [
        # Two inputs (param, const)
        (np.float16, torch.bfloat16),
        (np.float32, torch.bfloat16),
        (np.float32, torch.int32),
        (np.int32, torch.float32),
        (np.float16, torch.float32),
        (np.float16, torch.float32),
        (np.int16, torch.float16),
        (np.int32, torch.int64),
        # Three inputs (param, const, const)
        (np.float32, torch.int32, torch.int32),
        (np.float32, torch.int32, torch.float32),
        (np.int32, torch.float32, torch.int32),
        (np.float32, torch.int32, torch.int16),
        (np.int32, torch.float32, torch.int16),
        (np.int16, torch.int32, torch.int16),
        (np.float16, torch.float32, torch.float16),
        (np.float32, torch.float16, torch.float32),
        (np.float16, torch.int32, torch.int16),
        (np.int16, torch.float16, torch.int16),
        (np.float16, torch.bfloat16, torch.float32),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_cat(self, ie_device, precision, ir_version, in_types, trace_model):
        self._test(self.create_model_param_first(in_types), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model)

    @pytest.mark.parametrize(("in_types"), [
        # Three inputs (param, const, const)
        (np.float16, torch.int32, torch.int16),
        (np.int16, torch.float16, torch.int16),
        (np.float16, torch.bfloat16, torch.float32),
        (np.float16, torch.float32, torch.bfloat16),
        (np.float32, torch.bfloat16, torch.bfloat16),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_cat_param_mid(self, ie_device, precision, ir_version, in_types, trace_model):
        self._test(self.create_model_param_mid(in_types), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model)

    @pytest.mark.parametrize(("in_types"), [
        # Three inputs (param, const, const)
        (np.float16, torch.int32, torch.int16),
        (np.int16, torch.float16, torch.int16),
        (np.float16, torch.bfloat16, torch.float32),
        (np.float16, torch.float32, torch.bfloat16),
        (np.float32, torch.bfloat16, torch.bfloat16),
    ])
    @pytest.mark.parametrize(("trace_model"), [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_cat_param_last(self, ie_device, precision, ir_version, in_types, trace_model):
        self._test(self.create_model_param_last(in_types), None, ["aten::cat", "prim::ListConstruct"],
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types}, trace_model=trace_model)

# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.utils import get_models_list
from openvino import convert_model


def flattenize_tuples(list_input):
    if not isinstance(list_input, (tuple, list)):
        return [list_input]
    unpacked_pt_res = []
    for r in list_input:
        unpacked_pt_res.extend(flattenize_tuples(r))
    return unpacked_pt_res


def flattenize_structure(outputs):
    if not isinstance(outputs, dict):
        outputs = flattenize_tuples(outputs)
        return [i.numpy(force=True) if isinstance(i, torch.Tensor) else i for i in outputs]
    else:
        return dict((k, v.numpy(force=True) if isinstance(v, torch.Tensor) else v) for k, v in outputs.items())


def process_pytest_marks(filepath: str):
    return [
        pytest.param(n, marks=pytest.mark.xfail(reason=r) if m == "xfail" else pytest.mark.skip(reason=r)) if m else n
        for n, _, m, r in get_models_list(filepath)]


class TestTorchConvertModel(TestConvertModel):
    def setup_class(self):
        torch.set_grad_enabled(False)

    def load_model(self, model_name, model_link):
        raise "load_model is not implemented"

    def get_inputs_info(self, model_obj):
        return None

    def prepare_inputs(self, inputs_info):
        inputs = getattr(self, "inputs", self.example)
        if isinstance(inputs, dict):
            return dict((k, v.numpy()) for k, v in inputs.items())
        else:
            return [i.numpy() for i in inputs]

    def convert_model(self, model_obj):
        ov_model = convert_model(
            model_obj, example_input=self.example, verbose=True)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        if isinstance(inputs, dict):
            inps = dict((k, torch.from_numpy(v)) for k, v in inputs.items())
            fw_outputs = model_obj(**inps)
        else:
            fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        return flattenize_structure(fw_outputs)

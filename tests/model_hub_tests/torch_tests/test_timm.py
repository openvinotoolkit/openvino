# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import timm
import torch
import pytest
from models_hub_common.test_convert_model import TestConvertModel
from models_hub_common.constants import hf_hub_cache_dir
from models_hub_common.utils import cleanup_dir
from openvino import convert_model


def filter_timm(timm_list: list) -> list:
    unique_models = set()
    filtered_list = []
    ignore_set = {"base", "mini", "small", "xxtiny", "xtiny", "tiny", "lite", "nano", "pico", "medium", "big",
                  "large", "xlarge", "xxlarge", "huge", "gigantic", "giant", "enormous", "xs", "xxs", "s", "m", "l", "xl"}
    for name in timm_list:
        # first: remove datasets
        name_parts = name.split(".")
        _name = "_".join(name.split(".")[:-1]) if len(name_parts) > 1 else name
        # second: remove sizes
        name_set = set([n for n in _name.split("_") if not n.isnumeric()])
        name_set = name_set.difference(ignore_set)
        name_join = "_".join(name_set)
        if name_join not in unique_models:
            unique_models.add(name_join)
            filtered_list.append(name)
    return filtered_list


def get_all_models() -> list:
    m_list = timm.list_pretrained()
    return filter_timm(m_list)


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTimmConvertModel(TestConvertModel):
    def load_model(self, model_name, model_link):
        m = timm.create_model(model_name, pretrained=True)
        cfg = timm.get_pretrained_cfg(model_name)
        shape = [1] + list(cfg.input_size)
        self.example = (torch.randn(shape),)
        self.inputs = (torch.randn(shape),)
        return m

    def get_inputs_info(self, model_obj):
        return None

    def prepare_inputs(self, inputs_info):
        return [i.numpy() for i in self.inputs]

    def convert_model(self, model_obj):
        ov_model = convert_model(model_obj, example_input=self.example)
        return ov_model

    def infer_fw_model(self, model_obj, inputs):
        fw_outputs = model_obj(*[torch.from_numpy(i) for i in inputs])
        if isinstance(fw_outputs, dict):
            for k in fw_outputs.keys():
                fw_outputs[k] = fw_outputs[k].numpy(force=True)
        elif isinstance(fw_outputs, (list, tuple)):
            fw_outputs = [o.numpy(force=True) for o in fw_outputs]
        else:
            fw_outputs = [fw_outputs.numpy(force=True)]
        return fw_outputs

    def teardown_method(self):
        # remove all downloaded files from cache
        cleanup_dir(hf_hub_cache_dir)
        super().teardown_method()

    @pytest.mark.parametrize("name", ["mobilevitv2_050.cvnets_in1k",
                                      "poolformerv2_s12.sail_in1k",
                                      "vit_base_patch8_224.augreg_in21k",
                                      "beit_base_patch16_224.in22k_ft_in22k"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, ie_device):
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("name", get_all_models())
    def test_convert_model_all_models(self, name, ie_device):
        self.run(name, None, ie_device)

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import timm
import torch
from models_hub_common.utils import get_models_list, retry

from torch_utils import TestTorchConvertModel


def filter_timm(timm_list: list) -> list:
    unique_models = {}
    size_tokens = [
        "zepto", "atto", "femto", "pico", "nano", "micro", "xxtiny", "xxsmall",
        "xxs", "xtiny", "xsmall", "xs", "tiny", "s", "mini", "small", "lite",
        "medium", "m", "base", "big", "large", "l", "xlarge", "xl", "xxlarge",
        "huge", "gigantic", "giant", "enormous",
    ]
    size_order = {token: idx for idx, token in enumerate(size_tokens)}
    size_aliases = {
        "mediumd": "medium",
        "minimal": "mini",
        "giantopt": "giant",
        "xx": "xxs",
    }

    def parse_numeric_size(token: str) -> int | None:
        suffix_map = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
        if len(token) > 1 and token[:-1].isdigit() and token[-1] in suffix_map:
            return int(token[:-1]) * suffix_map[token[-1]]
        return None

    def normalize_size_token(token: str) -> str | None:
        token = size_aliases.get(token, token)
        if token in size_order:
            return token
        trimmed_digits = token.rstrip("0123456789")
        if trimmed_digits and trimmed_digits in size_order:
            return trimmed_digits
        return None

    for original_name in sorted(timm_list):
        normalized_name = original_name
        if "xx_small" in normalized_name or "x_small" in normalized_name:
            # x_small or xx_small should be merged to xsmall and xxsmall for canonical comparison only
            normalized_name = normalized_name.replace("xx_small", "xxsmall").replace("x_small", "xsmall")
        name_parts = normalized_name.split(".")
        base_name = "_".join(name_parts[:-1]) if len(name_parts) > 1 else normalized_name
        size_rank = (2, float("inf"))
        size_token_count = 0
        tokens = []
        for token in base_name.split("_"):
            if not token or token.isnumeric():
                continue
            lowered = token.lower()
            numeric_size = parse_numeric_size(lowered)
            if numeric_size is not None:
                size_rank = min(size_rank, (0, float(numeric_size)))
                size_token_count += 1
                continue
            normalized = normalize_size_token(lowered)
            if normalized is not None:
                idx = size_order[normalized]
                size_rank = min(size_rank, (1, float(idx)))
                size_token_count += 1
                continue
            tokens.append(lowered)

        name_join = "_".join(sorted(tokens))
        candidate = (size_rank, size_token_count, original_name)
        current = unique_models.get(name_join)
        if current is None or candidate < current:
            unique_models[name_join] = candidate
    return sorted(value[2] for value in unique_models.values())


# To make tests reproducible we seed the random generator
torch.manual_seed(0)


class TestTimmConvertModel(TestTorchConvertModel):
    @retry(3, exceptions=(OSError,), delay=5)
    def load_model(self, model_name, model_link):
        m = timm.create_model(model_name, pretrained=True)
        cfg = timm.get_pretrained_cfg(model_name)
        shape = list(cfg.input_size)
        self.example = (torch.randn([2] + shape),)
        self.inputs = (torch.randn([3] + shape),)
        if getattr(self, "mode", None) == "export":
            batch = torch.export.Dim("batch", min=1, max=3)
            self.export_kwargs = {"dynamic_shapes": {"x": {0: batch}}}
        return m

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

    @pytest.mark.parametrize("name", ["mobilevitv2_050.cvnets_in1k",
                                      "poolformerv2_s12.sail_in1k",
                                      "vit_base_patch8_224.augreg_in21k",
                                      "beit_base_patch16_224.in22k_ft_in22k",
                                      "sequencer2d_l.in1k",
                                      "gcresnext26ts.ch_in1k",
                                      "volo_d2_224.sail_in1k"])
    @pytest.mark.precommit
    def test_convert_model_precommit(self, name, ie_device):
        self.mode = "trace"
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    @pytest.mark.parametrize("name,link,mark,reason", get_models_list(os.path.join(os.path.dirname(__file__), "timm_models")))
    @pytest.mark.parametrize("mode", ["trace", "export"])
    def test_convert_model_all_models(self, mode, name, link, mark, reason, ie_device):
        self.mode = mode
        assert mark is None or mark in [
            'skip', 'xfail', 'xfail_trace', 'xfail_export'], f"Incorrect test case for {name}"
        if mark == 'skip':
            pytest.skip(reason)
        elif mark in ['xfail', f'xfail_{mode}']:
            pytest.xfail(reason)
        self.run(name, None, ie_device)

    @pytest.mark.nightly
    def test_models_list_complete(self, ie_device):
        m_list = timm.list_pretrained()
        all_models_ref = set(filter_timm(m_list))
        all_models = set([m for m, _, _, _ in get_models_list(
            os.path.join(os.path.dirname(__file__), "timm_models"))])
        assert all_models == all_models_ref, f"Lists of models are not equal."

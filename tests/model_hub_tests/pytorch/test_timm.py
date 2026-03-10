# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import re

import pytest
import timm
import torch
from models_hub_common.utils import get_models_list, retry

from torch_utils import TestTorchConvertModel


def filter_timm(timm_list: list) -> list:
    size_tokens = {
        "zepto", "atto", "femto", "pico", "nano", "micro", "xxtiny", "xxsmall",
        "xxs", "xtiny", "xsmall", "xs", "tiny", "s", "mini", "small", "lite",
        "medium", "m", "base", "big", "large", "l", "xlarge", "xl", "xxlarge",
        "huge", "gigantic", "giant", "enormous",
    }
    size_order = {token: idx for idx, token in enumerate(sorted(size_tokens))}
    size_aliases = {
        "mediumd": "medium",
        "minimal": "mini",
        "giantopt": "giant",
        "xx": "xxs",
    }
    resolution_pattern = re.compile(r"^(?:r)?(\d{2,4})(?:p)?$")
    prefixed_size_pattern = re.compile(r"^([a-z]{1,3})(\d{1,3})$")
    operation_hint_substrings = (
        "bias", "bn", "gn", "ln", "gap", "cls", "dw", "fused", "mlp",
        "rope", "attn", "msa", "mha", "retro", "stem", "patch", "token",
        "shift", "gated",
    )
    size_prefixes = {
        "b", "l", "m", "s", "t", "x", "n", "h", "w", "g", "p",
        "xl", "xx", "xs", "xt",
    }

    def split_tokens(*names: str | None) -> list[str]:
        tokens = []
        for name in names:
            if not name:
                continue
            normalized = name.replace("xx_small", "xxsmall").replace("x_small", "xsmall")
            normalized = normalized.replace('-', '_').replace('/', '_').lower()
            tokens.extend(token for token in normalized.split("_") if token)
        return tokens

    def is_size_like(token: str) -> bool:
        token = size_aliases.get(token, token)
        if token in size_tokens:
            return True
        if token.isdigit() or resolution_pattern.match(token):
            return True
        match = prefixed_size_pattern.match(token)
        if match and match.group(1) in size_prefixes:
            return not any(hint in token for hint in operation_hint_substrings)
        return False

    def architecture_signature(cfg, model_name: str) -> str:
        base_name = model_name.split(".")[0]
        arch_tokens = split_tokens(
            getattr(cfg, "architecture", None) or base_name,
            getattr(cfg, "architecture_tag", None),
            (getattr(cfg, "meta", None) or {}).get("variant") if cfg else None,
        )
        fallback = arch_tokens or split_tokens(base_name)
        filtered = [size_aliases.get(tok, tok) for tok in arch_tokens if not is_size_like(tok)]
        canonical = filtered or fallback
        unique = list(dict.fromkeys(canonical))  # preserve order
        return "_".join(unique) if unique else base_name.lower()

    def size_rank_from_name(model_name: str) -> tuple[float, float]:
        rank = (2.0, float("inf"))
        for token in split_tokens(model_name.split(".")[0]):
            normalized = size_aliases.get(token, token)
            if normalized in size_order:
                rank = min(rank, (0.0, float(size_order[normalized])))
                continue
            match = prefixed_size_pattern.match(normalized)
            if match and match.group(1) in size_prefixes:
                rank = min(rank, (1.0, float(match.group(2))))
        return rank

    selected = {}
    for original_name in sorted(timm_list):
        try:
            cfg = timm.get_pretrained_cfg(original_name)
        except Exception:
            cfg = None

        arch_key = architecture_signature(cfg, original_name)
        input_size = cfg.input_size[-1] if cfg and getattr(cfg, "input_size", None) else float("inf")
        candidate_rank = (
            size_rank_from_name(original_name),
            float(input_size),
            len(original_name),
            original_name,
        )

        current = selected.get(arch_key)
        if current is None or candidate_rank < current[0]:
            selected[arch_key] = (candidate_rank, original_name)

    return sorted(value[1] for value in selected.values())


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

    def get_supported_precommit_models():
        models = [
            "mobilevitv2_050.cvnets_in1k",
            "poolformerv2_s12.sail_in1k",
        ]
        if platform.machine() not in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64']:
            models.extend([
                "vit_tiny_patch16_224.augreg_in21k",
                "efficientnet_b0.ra_in1k",
                "convnext_atto.d2_in1k",
                "gcresnext26ts.ch_in1k",
                "volo_d1_224.sail_in1k",
            ])
        return models

    @pytest.mark.parametrize("name", get_supported_precommit_models())
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

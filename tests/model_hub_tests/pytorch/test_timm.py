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
    # ordered from smallest to largest
    size_list = [
        "zepto", "atto", "femto", "pico", "nano", "micro", "xxtiny", "xxsmall",
        "xxs", "xtiny", "xsmall", "xs", "tiny", "t", "s", "mini", "small", "lite",
        "medium", "m", "base", "big", "large", "l", "xlarge", "xl", "xxlarge",
        "huge", "h", "gigantic", "giant", "enormous",
    ]
    size_tokens = set(size_list)
    # size only if the name has no other size token, e.g. "iformer_h" but not "regnetz_040_h"
    contextual_size_tokens = {"t", "h"}
    size_order = {token: idx for idx, token in enumerate(size_list)}
    size_aliases = {
        "mediumd": "medium",
        "minimal": "mini",
        "giantopt": "giant",
        "xx": "xxs",
    }
    resolution_pattern = re.compile(r"^(?:r)?(\d{2,4})(?:p)?$")
    prefixed_size_pattern = re.compile(r"^([a-z]{1,3})(\d{1,3})(?:pt\d+)?$")
    # parameter count, e.g. "300m", "7b", "so400m"
    param_count_pattern = re.compile(r"^(?:so)?(\d+)([mb])$")
    # depth/cardinality x width multiplier, e.g. "50x1", "152x4", "32x8d"
    width_multiplier_pattern = re.compile(r"^(\d+)x(\d+)d?$")
    operation_hint_substrings = (
        "bias", "bn", "gn", "ln", "gap", "cls", "dw", "fused", "mlp",
        "rope", "attn", "msa", "mha", "retro", "stem", "patch", "token",
        "shift", "gated",
    )
    size_prefixes = {
        "b", "l", "m", "s", "t", "x", "n", "h", "w", "g", "p", "f", "e",
        "xl", "xx", "xs", "xt",
    }

    def split_tokens(*names: str | None) -> list[str]:
        tokens = []
        for name in names:
            if not name:
                continue
            normalized = name.replace("xx_small", "xxsmall").replace("x_small", "xsmall")
            normalized = normalized.replace("tiny_vit", "tinyvit")
            # split width multiplier from depth, e.g. "resnet50x4_clip" -> "resnet50_x4_clip",
            # "resnet50_clip" -> "resnet50_x1_clip"
            normalized = re.sub(r"(resnet\d+)(?:x(\d+))?(?=_clip)",
                                lambda m: f"{m.group(1)}_x{m.group(2) or 1}", normalized)
            normalized = normalized.replace('-', '_').replace('/', '_').lower()
            tokens.extend(token for token in normalized.split("_") if token)
        return tokens

    def param_count_millions(token: str) -> float | None:
        match = param_count_pattern.match(token)
        if not match:
            return None
        return float(match.group(1)) * (1000.0 if match.group(2) == "b" else 1.0)

    def width_multiplier(token: str) -> float | None:
        match = width_multiplier_pattern.match(token)
        return float(match.group(1)) * float(match.group(2)) if match else None

    def is_size_like(token: str, allow_contextual: bool = True) -> bool:
        token = size_aliases.get(token, token)
        if token in contextual_size_tokens:
            return allow_contextual
        if token in size_tokens or param_count_millions(token) is not None:
            return True
        if width_multiplier(token) is not None:
            return True
        if token.isdigit() or resolution_pattern.match(token):
            return True
        match = prefixed_size_pattern.match(token)
        if match and match.group(1) in size_prefixes:
            return not any(hint in token for hint in operation_hint_substrings)
        return False

    def allows_contextual(tokens: list[str]) -> bool:
        return not any(is_size_like(tok, allow_contextual=False) for tok in tokens)

    def architecture_signature(cfg, model_name: str) -> str:
        base_name = model_name.split(".")[0]
        arch_tokens = split_tokens(
            getattr(cfg, "architecture", None) or base_name,
            getattr(cfg, "architecture_tag", None),
            (getattr(cfg, "meta", None) or {}).get("variant") if cfg else None,
        )
        fallback = arch_tokens or split_tokens(base_name)
        allow_contextual = allows_contextual(arch_tokens)
        filtered = [size_aliases.get(tok, tok) for tok in arch_tokens if not is_size_like(tok, allow_contextual)]
        canonical = filtered or fallback
        unique = list(dict.fromkeys(canonical))  # preserve order
        return "_".join(unique) if unique else base_name.lower()

    def size_rank_from_name(model_name: str) -> tuple[float, float]:
        rank = (2.0, float("inf"))
        tokens = split_tokens(model_name.split(".")[0])
        allow_contextual = allows_contextual(tokens)
        for token in tokens:
            normalized = size_aliases.get(token, token)
            if normalized in contextual_size_tokens and not allow_contextual:
                continue
            if normalized in size_order:
                rank = min(rank, (0.0, float(size_order[normalized])))
                continue
            params = param_count_millions(normalized)
            if params is not None:
                rank = min(rank, (1.5, params))
                continue
            width = width_multiplier(normalized)
            if width is not None:
                rank = min(rank, (1.5, width))
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
            from openvino import PartialShape
            self.dynamo_input = (PartialShape([-1] + shape),)
            # Use same batch as example because the FX decoder does not
            # fully propagate symbolic batch through reshape ops yet.
            self.inputs = (torch.randn([2] + shape),)
        if model_name.startswith("convit"):
            # convit caches rel_indices in the first forward, which makes tracing checks fail
            m(*self.example)
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

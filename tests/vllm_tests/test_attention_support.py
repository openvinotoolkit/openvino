# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify the plugin refuses attention settings its PagedAttention rewrite can't express.

The OV translator passes empty ALiBi slopes and sinks, has no soft-cap input,
and builds causal decoder attention only. A layer using any of those would
silently compute different attention, and once the plugin owns the KV-cache
write there is no falling back to vLLM's attention. So _own_kv_cache_update
must refuse such a model before touching any layer, and still take over a
plain decoder model.

No model download needed: stand-in layers on vLLM's real CPU backend class.
"""

from types import SimpleNamespace

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm import plugin


def _layer(**impl):
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    defaults = {"alibi_slopes": None, "sinks": None, "logits_soft_cap": 0, "attn_type": "decoder"}
    return SimpleNamespace(attn_backend=CPUAttentionBackend, impl=SimpleNamespace(**{**defaults, **impl}))


class _Model:
    def __init__(self, layers):
        self._layers = layers

    def named_modules(self):
        return [(f"layers.{i}.attn", layer) for i, layer in enumerate(self._layers)]

    def modules(self):
        return list(self._layers)


@pytest.mark.precommit
def test_plain_decoder_attention_is_taken_over():
    layers = [_layer(), _layer()]
    assert plugin._own_kv_cache_update(_Model(layers)) == 2
    assert all(layer.attn_backend.forward_includes_kv_cache_update for layer in layers)


@pytest.mark.precommit
@pytest.mark.parametrize("impl, feature", [
    ({"alibi_slopes": [0.5, 0.25]}, "ALiBi slopes"),
    ({"sinks": object()}, "attention sinks"),
    ({"logits_soft_cap": 50.0}, "logits soft-capping"),
    ({"attn_type": "encoder_only"}, "attention type"),
], ids=["alibi", "sinks", "soft_cap", "encoder_only"])
def test_unsupported_attention_is_refused_before_takeover(impl, feature):
    from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackend

    layers = [_layer(), _layer(**impl)]
    with pytest.raises(NotImplementedError, match=feature):
        plugin._own_kv_cache_update(_Model(layers))
    assert all(layer.attn_backend is CPUAttentionBackend for layer in layers), "a layer was taken over anyway"

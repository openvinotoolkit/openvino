# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Verify build_call_kwargs binds positional values to the non-__pa__ ports in order, and refuses a count mismatch.

The positional values come from the FX graph's arguments and bind by position, so if the converted model ever had
one input more or fewer than the values passed, every later port would silently take its neighbour's tensor. That
must fail loudly instead.

No model download needed: stand-in ports and side channels.
"""

import pytest

from openvino.frontend.pytorch.torchdynamo.vllm import runtime_hooks


class _Port:
    def __init__(self, name):
        self._name = name

    def get_names(self):
        return {self._name}


class _Compiled:
    def __init__(self, names):
        self.inputs = [_Port(name) for name in names]


@pytest.fixture
def side_channels(monkeypatch):
    monkeypatch.setattr(runtime_hooks, "_bind_paged_attention_side_channel",
                        lambda compiled: {"__pa__key_cache": "kv"})
    monkeypatch.setattr(runtime_hooks, "_bind_gdn_side_channel", lambda compiled: ({}, None))


@pytest.mark.precommit
def test_positional_values_bind_in_order(side_channels):
    compiled = _Compiled(["input_ids", "__pa__key_cache", "positions"])
    call_kwargs, _ = runtime_hooks.build_call_kwargs(compiled, ["ids", "pos"])
    assert [call_kwargs[port] for port in compiled.inputs] == ["ids", "kv", "pos"]


@pytest.mark.precommit
@pytest.mark.parametrize("values", [["ids"], ["ids", "pos", "extra"]])
def test_value_count_mismatch_raises(side_channels, values):
    compiled = _Compiled(["input_ids", "__pa__key_cache", "positions"])
    with pytest.raises(RuntimeError, match="positional inputs"):
        runtime_hooks.build_call_kwargs(compiled, values)

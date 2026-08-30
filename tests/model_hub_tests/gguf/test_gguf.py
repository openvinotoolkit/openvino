# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Verifies the GGUF frontend against real .gguf checkpoints downloaded from the Hugging Face
# Hub, one per architecture listed in src/frontends/gguf/docs/supported_models.md.
#
# Each test downloads one .gguf file, converts it with core.read_model(), compiles it, and
# runs a single forward step (one new token, no prior context -- see utils.py for why that is
# enough to exercise the frontend's default stateless lowering without extra setup). The
# check is deliberately shallow: a finite, correctly shaped logits tensor. It does not build a
# tokenizer or run more than one step, since this suite is about frontend conversion/inference
# correctness, not generation quality or text coherence (which
# src/frontends/gguf/tests/compare_with_llama.py and the GenAI-based harness described in
# supported_models.md already cover for the architectures where that matters).
#
# gguf_models_precommit lists the small (<=~4B parameter) architectures, run on every commit.
# gguf_models_nightly lists the rest (up to ~30B), run nightly given their download size.

import os

import pytest

from models_hub_common.constants import clean_hf_cache_dir, hf_cache_dir
from models_hub_common.utils import cleanup_dir
from utils import assert_valid_logits, parse_gguf_model_list, run_gguf_model


class TestGGUF:
    def teardown_method(self):
        if clean_hf_cache_dir:
            cleanup_dir(hf_cache_dir)

    def run(self, arch, repo_id, filename, mark, reason, ie_device):
        assert mark in (None, "skip", "xfail"), f"Unknown mark {mark!r} for {arch}"
        if mark == "skip":
            pytest.skip(reason)
        if mark == "xfail":
            pytest.xfail(reason)

        logits = run_gguf_model(repo_id, filename, device=ie_device)
        assert_valid_logits(logits)

    @pytest.mark.parametrize(
        "arch,repo_id,filename,mark,reason",
        parse_gguf_model_list(os.path.join(os.path.dirname(__file__), "gguf_models_precommit")))
    @pytest.mark.precommit
    def test_gguf_precommit(self, arch, repo_id, filename, mark, reason, ie_device):
        self.run(arch, repo_id, filename, mark, reason, ie_device)

    @pytest.mark.parametrize(
        "arch,repo_id,filename,mark,reason",
        parse_gguf_model_list(os.path.join(os.path.dirname(__file__), "gguf_models_nightly")))
    @pytest.mark.nightly
    def test_gguf_nightly(self, arch, repo_id, filename, mark, reason, ie_device):
        self.run(arch, repo_id, filename, mark, reason, ie_device)

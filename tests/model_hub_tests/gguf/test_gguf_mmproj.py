# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Verifies the GGUF frontend's multimodal projector conversion against real mmproj-*.gguf files
# from the Hugging Face Hub. Each test converts the file, compiles every encoder it declares, and
# compares the embeddings with llama.cpp's CPU encoder, built at a pinned revision during the test
# session (see llama_cpp_oracle.py). llama.cpp runs an F32 copy of the same represented weights,
# so the comparison measures conversion and execution, not the file's quantization.
#
# Encoders with dynamic input sizes run twice on one compiled model, at two grid shapes.
#
# To check local files, list them with an empty repo_id and an absolute filename in a file of
# the same format, and point GGUF_MMPROJ_LIST at it; it replaces the precommit list.

import os
import sys

import numpy as np
import openvino as ov
import pytest
from openvino.frontend import FrontEndManager

from models_hub_common.constants import clean_hf_cache_dir, hf_cache_dir
from models_hub_common.utils import cleanup_dir
from llama_cpp_oracle import build_mmproj_oracle, frontend_test_file, llama_cpp_checkout
from utils import gguf_hub_download, parse_gguf_model_list

# Input builders and the dequantizer are shared with src/frontends/gguf/tests.
sys.path.insert(0, str(frontend_test_file("mmproj_fixtures.py").parent))
from mmproj_fixtures import DYNAMIC_VISION, checkpoint_inputs, grid_unit, run_oracle  # noqa: E402

# CPU runs in F32 like the frontend's accuracy suites; other devices are not yet calibrated.
NMSE_LIMIT = {"CPU": 1e-5}
DEFAULT_NMSE_LIMIT = 1e-4


@pytest.fixture(scope="module")
def oracle():
    # The pinned checkout's gguf-py writes the F32 reference copies.
    sys.path.insert(0, str(llama_cpp_checkout() / "gguf-py"))
    return build_mmproj_oracle()


def encoder(model, modality):
    """The subgraph that computes one modality's embeddings, with only its own inputs."""
    output = model.output(modality + ".embeddings")
    reachable, pending = set(), [output.get_node()]
    while pending:
        node = pending.pop()
        if node not in reachable:
            reachable.add(node)
            pending += [value.get_node() for value in node.input_values()]
    return ov.Model([output], [p for p in model.get_parameters() if p in reachable])


def input_sizes(metadata, modality):
    """One or two (width, height) shapes the encoder accepts; audio uses (frames, None)."""
    projector = metadata[modality + ".projector"]
    unit = grid_unit(metadata, modality)
    if modality == "audio":
        frames = {"gemma4ua": (9, 13), "gemma4a": (101, 152)}.get(projector, (16 * unit, 25 * unit))
        return [(count, None) for count in frames]
    if projector not in DYNAMIC_VISION:
        return [(None, None)]
    return [(8 * unit, 4 * unit), (4 * unit, 6 * unit)]


def nmse(actual, expected):
    error = actual.astype(np.float64) - expected
    energy = float(np.dot(expected.astype(np.float64), expected))
    return float(np.dot(error, error)) / energy if energy > 0 else float("inf")


class TestGGUFMMProj:
    def teardown_method(self):
        if clean_hf_cache_dir:
            cleanup_dir(hf_cache_dir)

    def run(self, projector, repo_id, filename, mark, reason, ie_device, oracle, tmp_path):
        limit = NMSE_LIMIT.get(ie_device, DEFAULT_NMSE_LIMIT)
        if mark and mark.startswith("nmse="):
            limit, mark = float(mark[len("nmse="):]), None
        assert mark in (None, "skip", "xfail"), f"Unknown mark {mark!r} for {projector}"
        if mark == "skip":
            pytest.skip(reason)
        if mark == "xfail":
            pytest.xfail(reason)
        from dequantize_mmproj import dequantize_mmproj

        path = gguf_hub_download(repo_id, filename) if repo_id else filename
        frontend = FrontEndManager().load_by_framework("gguf")
        model = frontend.convert(frontend.load(path))
        metadata = model.get_rt_info(["gguf_mmproj"]).value
        modalities = [m for m in ("vision", "audio") if m + ".projector" in metadata]
        assert "+".join(metadata[m + ".projector"] for m in modalities) == projector
        reference = tmp_path / "reference-F32.gguf"
        dequantize_mmproj(path, reference)
        try:
            self.compare(model, metadata, modalities, projector, reference, oracle, ie_device, limit)
        finally:
            reference.unlink()

    @staticmethod
    def compare(model, metadata, modalities, projector, reference, oracle, ie_device, limit):
        core = ov.Core()
        for modality in modalities:
            request = core.compile_model(encoder(model, modality), ie_device, {
                "INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0,
            }).create_infer_request()
            for width, height in input_sizes(metadata, modality):
                feeds, raw, width, height = checkpoint_inputs(metadata, modality, width, height)
                expected = run_oracle(oracle, reference, modality, width, height, raw)
                request.infer(feeds)
                actual = request.get_output_tensor().data.reshape(-1)
                label = f"{projector} {modality} {width}x{height}"
                assert actual.shape == expected.shape, f"{label}: {actual.shape} vs llama.cpp {expected.shape}"
                assert np.isfinite(actual).all(), f"{label}: non-finite embeddings"
                error = nmse(actual, expected)
                print(f"{label}: normalized MSE {error:.3g} (limit {limit})")
                assert error < limit, f"{label}: normalized MSE {error:.3g} against llama.cpp exceeds {limit}"

    @pytest.mark.parametrize(
        "projector,repo_id,filename,mark,reason",
        parse_gguf_model_list(
            os.environ.get("GGUF_MMPROJ_LIST", os.path.join(os.path.dirname(__file__), "gguf_mmproj_precommit"))))
    @pytest.mark.precommit
    def test_gguf_mmproj_precommit(self, projector, repo_id, filename, mark, reason, ie_device, oracle, tmp_path):
        self.run(projector, repo_id, filename, mark, reason, ie_device, oracle, tmp_path)

    @pytest.mark.parametrize(
        "projector,repo_id,filename,mark,reason",
        parse_gguf_model_list(os.path.join(os.path.dirname(__file__), "gguf_mmproj_nightly")))
    @pytest.mark.nightly
    def test_gguf_mmproj_nightly(self, projector, repo_id, filename, mark, reason, ie_device, oracle, tmp_path):
        self.run(projector, repo_id, filename, mark, reason, ie_device, oracle, tmp_path)

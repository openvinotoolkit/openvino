# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""llama.cpp CPU oracle fixtures for GGUFMMProjAccuracy (one input grid) and
GGUFMMProjDynamicAccuracy (two grids, prefixed 0. and 1., served by one compiled model).

PYTHONPATH=<llama.cpp>/gguf-py python gen_mmproj_accuracy.py --oracle <mmproj_oracle> \
    [--ops-oracle <mmproj_ops_oracle>] [--families ...]
See test_data/mmproj_accuracy/README.md for the reference revision. --ops-oracle also
regenerates the standalone op expectations.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np

from mmproj_fixtures import TensorWriter, finish, merge_window_order, muse_glimmer_indices, save_npz, split_variants

SINGLE_GRID = ("gemma3", "gemma3_fused", "gemma3_legacy", "idefics3", "janus_pro", "mlp", "mlp_norm", "mlp_feature",
               "qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma", "qwen2vl_merger",
               "qwen2.5vl_merger", "qwen3vl_merger", "internvl", "internvl_qknorm", "qwen2.5vl_merger_window_video",
               "voxtral_odd", "resampler", "resampler_v2", "resampler_v4")
# Single-grid models rerun here on two grids with different token counts.
REGRIDDED = ("qwen2.5vl_merger_grids", "resampler_grids")
TWO_GRID = ("muse-glimmer", "pixtral", "pixtral_merge", "phi4", "gemma4v", "gemma4uv", "gemma4uv_low_contrast",
            "gemma4ua", "gemma4v_one_sided", "minicpmv4_6", "gemma4a", "deepseekocr", "deepseekocr2",
            "deepseekocr_resize", "deepseekocr_overview", "deepseekocr2_overview", *REGRIDDED)


def write_single_grid_model(path, projector, projection_width=6):
    projector, _, variant = projector.partition("_") if projector.startswith("gemma3_") else (projector, "", "")
    # Concatenated CLIP feature layers; InternViT-style whole-tensor QK norms.
    if projector in ("mlp_feature", "internvl_qknorm"):
        projector, _, variant = projector.partition("_")
    if projector.startswith("resampler"):
        projector, _, variant = projector.partition("_")
        projection_width = 128  # one 128-wide cross-attention head
    audio = projector in {"qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma"}
    clip = projector in {"mlp", "mlp_norm"}
    qwen = projector in {"qwen2vl_merger", "qwen2.5vl_merger", "qwen3vl_merger"}
    modality, prefix = ("audio", "a.") if audio else ("vision", "v.")
    key = f"clip.{modality}."
    w = gguf.GGUFWriter(path, "clip")
    w.add_bool(f"clip.has_{modality}_encoder", True)
    w.add_string("clip.projector_type", "mlp" if projector == "mlp_norm" else projector)
    w.add_bool("clip.use_gelu", True)
    if projector == "resampler":
        if variant == "v4":
            w.add_int32("clip.minicpmv_version", 4)
        elif not variant:
            w.add_int32("clip.minicpmv_version", 3)
            w.add_uint32("clip.minicpmv_query_num", 3)
    for name, value in {"embedding_length": 8, "feed_forward_length": 12, "block_count": 2,
                        "projection_dim": projection_width, "attention.head_count": 2}.items():
        w.add_uint32(key + name, value)
    w.add_float32(key + "attention.layer_norm_epsilon", 1e-5)
    if variant == "feature":
        w.add_array(key + "feature_layer", [1, 2])
    rng = np.random.default_rng(1729)

    def tensor(name, shape, norm=False):
        if projector == "internvl" and name.startswith("mm."):
            name = name.replace("mm.", "mm.model.mlp.", 1)
        if variant == "legacy":
            name = name.replace("ffn_up", "ffn_tmp").replace("ffn_down", "ffn_up").replace("ffn_tmp", "ffn_down")
        values = rng.normal(0, 0.08, shape).astype(np.float32)
        w.add_tensor(name, values + 1 if norm else values)

    tensor(prefix + "position_embd.weight", (4900 if projector == "resampler" else
                                            17 if clip or projector == "internvl" else 16, 8))
    if clip or projector == "internvl":
        tensor("v.class_embd", (1, 8))
    for i in range(2):
        p = prefix + f"blk.{i}."
        for name in ("ln1", "ln2"):
            tensor(p + name + ".weight", (8,), True)
            tensor(p + name + ".bias", (8,))
        for name in (("attn_out",) if variant == "fused" else ("attn_q", "attn_k", "attn_v", "attn_out")):
            tensor(p + name + ".weight", (8, 8))
            if not (audio and name == "attn_k"):
                tensor(p + name + ".bias", (8,))
        if variant == "fused":
            tensor(p + "attn_qkv.weight", (24, 8))
            tensor(p + "attn_qkv.bias", (24,))
            tensor(p + "attn_q_norm.weight", (4,), True)
            tensor(p + "attn_k_norm.weight", (4,), True)
            tensor(p + "ls1.weight", (8,))
            tensor(p + "ls2.weight", (8,))
            tensor(p + "out_scale.weight", (8,), True)
            tensor(p + "attn_post_norm.weight", (8,), True)
            tensor(p + "ffn_post_norm.weight", (8,), True)
        if variant == "qknorm":
            tensor(p + "attn_q_norm.weight", (8,), True)
            tensor(p + "attn_k_norm.weight", (8,), True)
        tensor(p + "ffn_up.weight", (12, 8))
        tensor(p + "ffn_up.bias", (12,))
        tensor(p + "ffn_down.weight", (8, 12))
        tensor(p + "ffn_down.bias", (8,))
        if projector == "qwen3vl_merger":
            # The reference requires a fused projection for Qwen3 VL.
            tensor(p + "attn_qkv.weight", (24, 8))
            tensor(p + "attn_qkv.bias", (24,))
    if projector == "qwen3vl_merger":
        tensor("v.deepstack.0.norm.weight", (32,), True)
        tensor("v.deepstack.0.norm.bias", (32,))
        tensor("v.deepstack.0.fc1.weight", (12, 32))
        tensor("v.deepstack.0.fc1.bias", (12,))
        tensor("v.deepstack.0.fc2.weight", (6, 12))
        tensor("v.deepstack.0.fc2.bias", (6,))
    tensor(prefix + "post_ln.weight", (8,), True)
    tensor(prefix + "post_ln.bias", (8,))
    if not audio:
        w.add_uint32(key + "image_size", 8)
        w.add_uint32(key + "patch_size", 2)
        w.add_uint32(key + "projector.scale_factor", 2)
        w.add_array(key + "image_mean", [0.5, 0.5, 0.5])
        w.add_array(key + "image_std", [0.5, 0.5, 0.5])
        tensor("v.patch_embd.weight", (8, 3, 2, 2))
        if not qwen or projector == "qwen3vl_merger":
            tensor("v.patch_embd.bias", (8,))
        if qwen:
            w.add_uint32(key + "spatial_merge_size", 2)
            w.add_uint32(key + "n_wa_pattern", 2 if projector == "qwen2.5vl_merger" else 0)
            tensor("v.patch_embd.weight.1", (8, 3, 2, 2))
        if projector == "resampler":
            queries = 96 if variant == "v2" else 64 if variant == "v4" else 3
            tensor("resampler.query", (queries, projection_width))
            tensor("resampler.pos_embed_k", (16, projection_width))
            tensor("resampler.proj.weight", (projection_width, projection_width))
            tensor("resampler.kv.weight", (projection_width, 8))
            for name in ("q", "k", "v", "out"):
                tensor(f"resampler.attn.{name}.weight", (projection_width, projection_width))
                tensor(f"resampler.attn.{name}.bias", (projection_width,))
            for name in ("q", "kv", "post"):
                tensor(f"resampler.ln_{name}.weight", (projection_width,), True)
                tensor(f"resampler.ln_{name}.bias", (projection_width,))
        elif projector == "gemma3":
            tensor("mm.soft_emb_norm.weight", (8,), True)
            tensor("mm.input_projection.weight", (8, projection_width))
        elif projector == "idefics3":
            tensor("mm.model.fc.weight", (6, 32))
        elif qwen:
            tensor("mm.0.weight", (12, 32))
            tensor("mm.0.bias", (12,))
            tensor("mm.2.weight", (6, 12))
            tensor("mm.2.bias", (6,))
        elif projector == "internvl":
            tensor("mm.0.weight", (32,), True)
            tensor("mm.0.bias", (32,))
            tensor("mm.1.weight", (12, 32))
            tensor("mm.1.bias", (12,))
            tensor("mm.3.weight", (6, 12))
            tensor("mm.3.bias", (6,))
        elif clip:
            tensor("mm.0.weight", (12, 16 if variant == "feature" else 8))
            tensor("mm.0.bias", (12,))
            tensor("mm.3.weight" if projector == "mlp_norm" else "mm.2.weight", (6, 12))
            tensor("mm.3.bias" if projector == "mlp_norm" else "mm.2.bias", (6,))
            if projector == "mlp_norm":
                tensor("mm.1.weight", (12,), True)
                tensor("mm.1.bias", (12,))
                tensor("mm.4.weight", (6,), True)
                tensor("mm.4.bias", (6,))
        else:
            tensor("mm.0.weight", (12, 8))
            tensor("mm.0.bias", (12,))
            tensor("mm.1.weight", (6, 12))
            tensor("mm.1.bias", (6,))
    else:
        w.add_uint32(key + "num_mel_bins", 4)
        w.add_uint32(key + "projector.stack_factor", 2)
        tensor("a.conv1d.1.weight", (8, 4, 3))
        tensor("a.conv1d.1.bias", (8, 1))
        tensor("a.conv1d.2.weight", (8, 8, 3))
        tensor("a.conv1d.2.bias", (8, 1))
        if projector == "qwen2a":
            tensor("mm.a.fc.weight", (6, 8))
            tensor("mm.a.fc.bias", (6,))
        elif projector == "meralion":
            tensor("mm.a.norm_pre.weight", (16,), True)
            tensor("mm.a.norm_pre.bias", (16,))
            for i, shape in enumerate(((12, 16), (12, 12), (12, 12), (6, 12))):
                tensor(f"mm.a.mlp.{i}.weight", shape)
                tensor(f"mm.a.mlp.{i}.bias", (shape[0],))
        else:
            dim = 8 if projector == "musicflamingo" else 16
            tensor("mm.a.mlp.1.weight", (24 if projector == "ultravox" else 12, dim))
            tensor("mm.a.mlp.2.weight", (6, 12))
            if projector == "ultravox":
                tensor("mm.a.norm_pre.weight", (16,), True)
                tensor("mm.a.norm_mid.weight", (12,), True)
            if projector == "musicflamingo":
                tensor("mm.a.mlp.1.bias", (12,))
                tensor("mm.a.mlp.2.bias", (6,))
            if projector == "glma":
                tensor("mm.a.norm_pre.weight", (8,), True)
                tensor("mm.a.norm_pre.bias", (8,))
                tensor("mm.a.mlp.1.bias", (12,))
                tensor("mm.a.mlp.2.bias", (6,))
                tensor("v.boi", (1, 6))
                tensor("v.eoi", (1, 6))
    finish(w)
    return audio


def single_grid_inputs(projector, width, height):
    """Raw oracle input, optional second video frame, encoder inputs and their family extras."""
    audio = projector.removesuffix("_odd") in {"qwen2a", "ultravox", "voxtral", "musicflamingo", "meralion", "glma"}
    window_video = projector.endswith("_window_video")
    qwen = "vl_merger" in projector
    shape = (height, width) if audio else (height, width, 3)
    raw = np.random.default_rng(42).normal(0, 0.4, shape).astype(np.float32)
    raw_second = np.random.default_rng(43).normal(0, 0.4, shape).astype(np.float32) if window_video else None
    inputs = raw.reshape(1, height, 1, width) if audio else raw.transpose(2, 0, 1)[None]
    extra = {}
    if projector.startswith("resampler"):
        rows, cols = np.indices((height // 2, width // 2))
        extra["position_h"] = rows.astype(np.float32).reshape(1, 1, -1, 1)
        extra["position_w"] = cols.astype(np.float32).reshape(1, 1, -1, 1)
        extra["position_ids"] = ((70 * rows // (height // 2)) * 70 +
                                  70 * cols // (width // 2)).astype(np.int32).reshape(1, 1, 1, -1)
    if qwen:
        inputs = np.repeat(inputs, 2, axis=0)
        indices = merge_window_order(height // 2, width // 2)
        rows, cols = np.divmod(indices, width // 2)
        extra["patch_indices"] = np.array(indices, np.int32).reshape(1, 1, 1, -1)
        extra["position_ids"] = np.array([rows, cols, rows, cols], np.int32).reshape(1, 1, 1, -1)
        extra["attention_mask"] = np.zeros((1, 1, len(indices), len(indices)), np.float32)
        extra["output_indices"] = np.arange(len(indices) // 4, dtype=np.int32).reshape(1, 1, 1, -1)
        if window_video:
            inputs[1] = raw_second.transpose(2, 0, 1)
            groups, windows = [], []
            gh, gw = height // 4, width // 4
            for y in range(0, gh, 28):
                for x in range(0, gw, 28):
                    window = [yy * gw + xx for yy in range(y, min(y + 28, gh))
                              for xx in range(x, min(x + 28, gw))]
                    groups.extend(window)
                    windows.append(len(window) * 4)
            permutation = np.array([4 * group + i for group in groups for i in range(4)])
            extra["patch_indices"] = extra["patch_indices"][..., permutation]
            extra["position_ids"] = np.array([rows[permutation], cols[permutation]] * 2,
                                               np.int32).reshape(1, 1, 1, -1)
            mask = np.full((len(indices), len(indices)), np.finfo(np.float32).min, np.float32)
            offset = 0
            for count in windows:
                mask[offset:offset + count, offset:offset + count] = 0
                offset += count
            extra["attention_mask"] = mask[None, None]
            extra["output_indices"] = np.argsort(groups).astype(np.int32).reshape(1, 1, 1, -1)
    return raw, raw_second, inputs, extra


def run_encoder_oracle(oracle, path, projector, width, height, raw, raw_second=None, env=None):
    """Flat mmproj_oracle embeddings for `raw` (and an optional second video frame)."""
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        raw.tofile(directory / "input.bin")
        second = []
        if raw_second is not None:
            raw_second.tofile(directory / "second.bin")
            second = [str(directory / "second.bin")]
        audio = raw.ndim == 2
        subprocess.run([str(oracle.resolve()), str(path), "audio" if audio else "vision",
                        str(width), str(height), str(directory / "input.bin"),
                        str(directory / "output.bin"), *second], check=True, env=env)
        return np.fromfile(directory / "output.bin", dtype=np.float32)


def single_grid_width(projector):
    return 128 if projector.startswith("resampler") else 12 if projector == "qwen3vl_merger" else 6


VARIANTS = ("_resize", "_overview", "_one_sided", "_low_contrast")
# mmproj_ops_oracle output file -> test_data .npy name.
OPS_EXPECTATIONS = {
    "window_input": "mmproj_window_input", "windows": "mmproj_windows", "restored": "mmproj_restored",
    "relative_table": "mmproj_relative_table", "relative": "mmproj_relative",
    "vision": "vision_rope_expected", "imrope": "multimodal_imrope_expected",
    "resize": "mmproj_interpolate_expected", "resize_corners": "mmproj_interpolate_corners_expected",
    "resize_down": "mmproj_interpolate_down_expected",
    "resize_down_corners": "mmproj_interpolate_down_corners_expected",
    "im2col7": "mmproj_im2col7", "im2col9": "mmproj_im2col9",
}


def write_muse_model(path):
    """Muse Glimmer: sparse/global windows, two-axis RoPE and pixel-shuffle merging."""
    w = gguf.GGUFWriter(path, "clip")
    t = TensorWriter(w, np.random.default_rng(20260922))
    tensor, linear = t.tensor, t.linear
    width, hidden, layers = 32, 48, 6
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "muse-glimmer")
    w.add_bool("clip.use_gelu", True)
    for name, value in {"embedding_length": width, "feed_forward_length": hidden,
                        "block_count": layers, "projection_dim": 12, "attention.head_count": 4,
                        "image_size": 4, "patch_size": 2, "spatial_merge_size": 2,
                        "image_min_pixels": 16, "image_max_pixels": 4096}.items():
        w.add_uint32("clip.vision." + name, value)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    w.add_array("clip.vision.image_mean", [.5, .5, .5])
    w.add_array("clip.vision.image_std", [.5, .5, .5])

    def norm(name):
        t.norm(name, width)

    tensor("v.patch_embd.weight", (width, 3, 2, 2))
    tensor("v.patch_embd.bias", (width,))
    tensor("v.position_embd.weight", (9, width))  # 3x3 windows; the fixture grids leave partial ones
    norm("v.pre_ln")
    norm("v.post_ln")
    for i in range(layers):
        p = f"v.blk.{i}."
        norm(p + "ln1")
        norm(p + "ln2")
        for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
            linear(p + name, width, width)
        linear(p + "ffn_up", width, hidden)
        linear(p + "ffn_down", hidden, width)
    linear("mm.0", width * 4, 24, False)
    linear("mm.1", 24, 20, False)
    linear("mm.2", 20, 12, False)
    finish(w)
    return 12


def write_grid_model(path, family):
    if family == "muse-glimmer":
        return write_muse_model(path)
    family, variants = split_variants(family, *VARIANTS)
    one_sided = "_one_sided" in variants
    low_contrast = "_low_contrast" in variants
    audio = family in {"gemma4ua", "gemma4a"}
    width, heads, hidden, output = 16, 2, 24, 12
    prefix, modality = ("a.", "audio") if audio else ("v.", "vision")
    w = gguf.GGUFWriter(path, "clip")
    t = TensorWriter(w, np.random.default_rng(2718))
    tensor, linear, norm = t.tensor, t.linear, t.norm

    w.add_bool(f"clip.has_{modality}_encoder", True)
    w.add_string("clip.projector_type", family.removesuffix("_merge"))
    w.add_bool("clip.use_gelu", True)
    key = f"clip.{modality}."
    for name, value in {"embedding_length": width, "feed_forward_length": hidden, "block_count": 0 if family in {"gemma4uv", "gemma4ua"} else 2,
                        "projection_dim": output, "attention.head_count": heads}.items():
        w.add_uint32(key + name, value)
    w.add_float32(key + "attention.layer_norm_epsilon", 1e-5)
    if family == "gemma4a":
        w.add_uint32(key + "num_mel_bins", 8)
        for i, (ci, co) in enumerate(((1, 4), (4, 8))):
            tensor(f"a.conv1d.{i}.weight", (co, ci, 3, 3))
            tensor(f"a.conv1d.{i}.bias", (co, 1, 1))
            norm(f"a.conv1d.{i}.norm", co, False)
        linear("a.input_projection", 16, width)
        for i in range(2):
            p = f"a.blk.{i}."
            for name in ("ffn_norm", "ffn_norm_1", "ffn_post_norm", "ffn_post_norm_1", "ln1", "ln2",
                         "attn_post_norm", "conv_norm", "norm_conv"):
                norm(p + name, width, False)
            for name in ("attn_q", "attn_k", "attn_v", "attn_out", "attn_k_rel", "conv_pw2"):
                linear(p + name, width, width, False)
            # Q-only input bounds catch clipping leaking into the shared K/V input.
            w.add_tensor(p + "attn_q.input_min", np.array([-.2], np.float32))
            w.add_tensor(p + "attn_q.input_max", np.array([.25], np.float32))
            norm(p + "per_dim_scale", width // heads, False)
            norm(p + "per_dim_k_scale", width // heads, False)
            for suffix in ("", "_1"):
                linear(p + "ffn_up" + suffix, width, hidden, False)
                linear(p + "ffn_down" + suffix, hidden, width, False)
            linear(p + "conv_pw1", width, width * 2, False)
            tensor(p + "conv_dw.weight", (width, 5))
            tensor(p + "conv_dw.bias", (width,))
        linear("a.pre_encode.out", width, width)
        norm("mm.a.soft_emb_norm", width, False)
        linear("mm.a.input_projection", width, output, False)
    elif audio:
        w.add_uint32(key + "num_mel_bins", 640)
        linear("mm.a.input_projection", 640, output, False)
    else:
        w.add_uint32(key + "image_size", 8)
        w.add_uint32(key + "patch_size", 24 if low_contrast else 2)
        w.add_uint32(key + "projector.scale_factor", 4 if family == "minicpmv4_6" else 2)
        w.add_uint32(key + "image_min_pixels", 16)
        w.add_uint32(key + "image_max_pixels", 4096)
        w.add_array(key + "image_mean", [.5, .5, .5])
        w.add_array(key + "image_std", [.5, .5, .5])
        if family.startswith("deepseekocr"):
            w.add_uint32(key + "sam.embedding_length", 8)
            w.add_uint32(key + "sam.head_count", 2)
            w.add_uint32(key + "sam.block_count", 3)
            w.add_uint32(key + "window_size", 7)
            if family == "deepseekocr2":
                w.add_uint32(key + "attention.head_count_kv", 1)
                tensor("v.resample_query_768.weight", (144, width))
                tensor("v.resample_query_1024.weight", (256, width))
            else:
                tensor("v.class_embd", (width,))
                tensor("v.position_embd.weight", (17, width))
                tensor("v.image_newline", (output,))
            tensor("v.view_seperator", (output,))
            tensor("v.sam.patch_embd.weight", (8, 3, 16, 16))
            tensor("v.sam.patch_embd.bias", (8,))
            tensor("v.sam.pos_embd.weight", (16, 16, 8))
            for i in range(12):  # pinned loader requires 12 sets even in a shallow fixture
                p = f"v.sam.blk.{i}."
                norm(p + "pre_ln", 8)
                norm(p + "post_ln", 8)
                linear(p + "attn.qkv", 8, 24)
                linear(p + "attn.out", 8, 8)
                tensor(p + "attn.pos_h.weight", (13, 4))
                tensor(p + "attn.pos_w.weight", (13, 4))
                linear(p + "mlp.lin1", 8, 12)
                linear(p + "mlp.lin2", 12, 8)
            tensor("v.sam.neck.0.weight", (8, 8, 1, 1))
            norm("v.sam.neck.1", 8)
            tensor("v.sam.neck.2.weight", (8, 8, 3, 3))
            norm("v.sam.neck.3", 8)
            tensor("v.sam.net_2.weight", (8, 8, 3, 3))
            tensor("v.sam.net_3.weight", (width, 8, 3, 3))
            linear("mm.model.fc", width if family == "deepseekocr2" else width * 2, output)
        if family == "gemma4uv":
            patch_width = 3 * (48 if low_contrast else 4) ** 2
            if low_contrast:
                # Large, nearly constant patch-projection rows amplify errors in
                # the mean of normalized low-contrast patches.
                weights = np.broadcast_to(np.linspace(-30, 30, width)[:, None] / patch_width,
                                          (width, patch_width)).copy()
                w.add_tensor("v.patch_embd.weight", weights.astype(np.float32))
                tensor("v.patch_embd.bias", (width,))
                w.add_tensor("v.patch_norm.1.weight", np.ones(patch_width, np.float32))
                w.add_tensor("v.patch_norm.1.bias", np.zeros(patch_width, np.float32))
            else:
                linear("v.patch_embd", patch_width, width)
                norm("v.patch_norm.1", patch_width)
            norm("v.patch_norm.2", width)
            norm("v.patch_norm.3", width)
        else:
            tensor("v.patch_embd.weight", (width, 3, 2, 2))
            if family != "gemma4v":
                tensor("v.patch_embd.bias", (width,))
        if family.startswith("gemma4"):
            tensor("v.position_embd.weight", (2, 32, width))
            linear("mm.input_projection", width, output, False)
            if family == "gemma4v":
                tensor("v.std_bias", (width,))
                tensor("v.std_scale", (width,), True)
        elif family == "phi4":
            tensor("v.position_embd.weight", (16, width))
            linear("mm.0", width, hidden)
            linear("mm.2", hidden, output)
        elif family.startswith("pixtral"):
            tensor("v.token_embd.img_break", (output,))
            linear("mm.1", width, hidden)
            linear("mm.2", hidden, output)
            if family.endswith("_merge"):
                w.add_uint32(key + "spatial_merge_size", 2)
                norm("mm.input_norm", width, False)
                linear("mm.patch_merger", width * 4, width, False)
        elif family == "minicpmv4_6":
            tensor("v.position_embd.weight", (4900, width))
            w.add_array(key + "wa_layer_indexes", [0])
            p = "v.vit_merger."
            norm(p + "ln1", width)
            for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
                linear(p + name, width, width)
            norm(p + "ds_ln", width * 4)
            linear(p + "ds_ffn_up", width * 4, hidden)
            linear(p + "ds_ffn_down", hidden, width)
            norm("mm.input_norm", width * 4)
            linear("mm.up", width * 4, hidden)
            linear("mm.down", hidden, output)
        if family != "gemma4uv":
            norm("v.pre_ln", width)
            norm("v.post_ln", width)
            for i in range(2):
                p = f"v.blk.{i}."
                norm(p + "ln1", width)
                norm(p + "ln2", width)
                for name in ("attn_q", "attn_k", "attn_v", "attn_out"):
                    linear(p + name, width, width // 2 if family == "deepseekocr2" and name in {"attn_k", "attn_v"} else width)
                if family == "gemma4v":
                    norm(p + "attn_post_norm", width, False)
                    norm(p + "ffn_post_norm", width, False)
                    norm(p + "attn_q_norm", width // heads, False)
                    norm(p + "attn_k_norm", width // heads, False)
                    # Nontrivial clipping catches omission or bias/clamp ordering errors.
                    for side in ("input", "output"):
                        if not one_sided or i == 0:
                            w.add_tensor(p + f"attn_q.{side}_min", np.array([-.2], np.float32))
                        if not one_sided or i == 1:
                            w.add_tensor(p + f"attn_q.{side}_max", np.array([.25], np.float32))
                linear(p + "ffn_up", width, hidden)
                linear(p + "ffn_down", hidden, width)
                if family.startswith("pixtral") or family in {"gemma4v", "deepseekocr2"}:
                    linear(p + "ffn_gate", width, hidden)
    finish(w)
    return output


def grid_inputs(family, width, height):
    if family == "muse-glimmer":
        raw = np.random.default_rng(42).uniform(-1, 1, (height, width, 3)).astype(np.float32)
        return raw, {"pixel_values": raw.transpose(2, 0, 1)[None], **muse_glimmer_indices(height // 2, width // 2, 3)}
    family, variants = split_variants(family, *VARIANTS)
    overview = "_overview" in variants
    raw = np.random.default_rng(42).normal(.1, .4, (height, width) if family in {"gemma4ua", "gemma4a"} else (height, width, 3)).astype(np.float32)
    if "_low_contrast" in variants:
        raw[:] = np.array([27, 29, 32], np.float32) / 255
    if family == "gemma4ua":
        return raw, {"waveform_frames": raw.T.reshape(1, 1, width, height)}
    if family == "gemma4a":
        return raw, {"features": raw.reshape(1, 1, height, width)}
    if family.startswith("deepseekocr"):
        # Repeat a nonzero tile to keep large SAM-grid fixtures compact on disk.
        raw = np.tile(np.random.default_rng(42).normal(.1, .4, (16, 16, 3)).astype(np.float32), (height // 16, width // 16, 1))
        side = width // 16
        local = np.arange(7)
        glob = np.arange(side)
        result = {"relative_indices_local": (local[:, None] - local[None] + 6).astype(np.int32)[None, None],
                  "relative_indices_global": (glob[:, None] - glob[None] + side - 1).astype(np.int32)[None, None]}
        if family == "deepseekocr2":
            n = (width // 64) ** 2
            result["pixel_values"] = raw.transpose(2, 0, 1)[None]
            result["query_indices"] = np.arange(0 if n == 144 else 144, 144 if n == 144 else 400, dtype=np.int32).reshape(1, 1, 1, -1)
            result["query_output_indices"] = np.arange(n, 2 * n, dtype=np.int32).reshape(1, 1, 1, -1)
            result["position_ids"] = np.arange(2 * n, dtype=np.int32).reshape(1, 1, 1, -1)
            q, k = np.indices((2 * n, 2 * n))
            result["attention_mask"] = np.where((k < n) | ((q >= n) & (k <= q)), 0, -1e9).astype(np.float32)[None, None]
            result["output_indices"] = np.arange(n + int(overview), dtype=np.int32).reshape(1, 1, 1, -1)
        else:
            batch = height // width
            result["pixel_values"] = raw.reshape(batch, width, width, 3).transpose(0, 3, 1, 2)
            grid = width // 64
            result["position_indices"] = np.arange(0 if grid == 4 else 17, 17 if grid == 4 else 18 + grid * grid, dtype=np.int32).reshape(1, 1, 1, -1)
            total = batch * grid * grid
            order = [index for y in range(grid) for index in
                     ([b * grid * grid + y * grid + x for b in range(batch) for x in range(grid)] + [total])]
            result["output_indices"] = np.array(order + ([total + 1] if overview else []), np.int32).reshape(1, 1, 1, -1)
        return raw, result
    result = {"pixel_values": raw.transpose(2, 0, 1)[None]}
    patch = (48 if "_low_contrast" in variants else 4) if family == "gemma4uv" else 2
    h, w = height // patch, width // patch
    rows, cols = np.indices((h, w))
    if family.startswith(("pixtral", "gemma4")):
        result.update(position_x=cols.astype(np.int32).reshape(1, 1, 1, -1),
                      position_y=rows.astype(np.int32).reshape(1, 1, 1, -1))
    if family == "minicpmv4_6":
        result["position_ids"] = (70 * (70 * rows // h) + 70 * cols // w).astype(np.int32).reshape(1, 1, 1, -1)
        order = merge_window_order(h, w)
        result["window_indices"] = np.array(order, np.int32).reshape(1, 1, 1, -1)
        result["inverse_window_indices"] = np.argsort(order).astype(np.int32).reshape(1, 1, 1, -1)
        index = np.arange(h * w) // 4
        result["attention_mask"] = np.where(index[:, None] == index[None], 0, np.finfo(np.float32).min).astype(np.float32)[None, None]
        for name, dh, dw in (("vit_merger", h, w), ("merger", h // 2, w // 2)):
            for i, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                result[f"{name}.indices.{i}"] = np.array([(y + dy) * dw + x + dx
                    for y in range(0, dh, 2) for x in range(0, dw, 2)], np.int32).reshape(1, 1, 1, -1)
    return raw, result


def single_grid_fixture(oracle, family):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.gguf"
        window_video = family.endswith("_window_video")
        audio = write_single_grid_model(path, family.removesuffix("_window_video").removesuffix("_odd"))
        qwen = "vl_merger" in family
        width, height = (120, 16) if window_video else (16, 4) if audio else (12, 8) if qwen else (8, 8)
        if family.startswith("resampler"):
            width, height = (12, 8) if family == "resampler" else (8, 12)
        if family.endswith("_odd"):
            width = 18
        raw, raw_second, inputs, extra = single_grid_inputs(family, width, height)
        output = run_encoder_oracle(oracle, path, family, width, height, raw, raw_second)
        output = output.reshape(1, 1, -1, single_grid_width(family))
        return {"model": np.frombuffer(path.read_bytes(), np.uint8), "inputs": inputs, "embeddings": output, **extra}


def two_grid_fixture(oracle, family):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.gguf"
        if family in REGRIDDED:
            projector = family.removesuffix("_grids")
            write_single_grid_model(path, projector)
            width_out = single_grid_width(projector)
            sizes = [(12, 8), (8, 16)]
        else:
            width_out = write_grid_model(path, family)
            sizes = {
                "deepseekocr2": [(768, 768), (1024, 1024)],
                "deepseekocr2_overview": [(768, 768), (1024, 1024)],
                "deepseekocr_resize": [(128, 128), (512, 512)],
                "deepseekocr_overview": [(256, 256), (256, 256)],
                "deepseekocr": [(256, 256), (256, 512)],
                "gemma4a": [(49, 8), (101, 8)],
                "gemma4ua": [(3, 640), (5, 640)],
                "gemma4uv_low_contrast": [(96, 48), (48, 144)],
                "muse-glimmer": [(12, 8), (4, 4)],
            }.get(family, [(16, 8), (8, 24)])
        environment = dict(os.environ, **({"GGUF_ORACLE_OVERVIEW": "1"} if family.endswith("_overview") else {}))
        fixtures = {"model": np.frombuffer(path.read_bytes(), np.uint8)}
        for step, (width, height) in enumerate(sizes):
            if family in REGRIDDED:
                raw, second, pixels, values = single_grid_inputs(projector, width, height)
                values = {"pixel_values": pixels, **values}
                embeddings = run_encoder_oracle(oracle, path, projector, width, height, raw, second)
            else:
                raw, values = grid_inputs(family, width, height)
                embeddings = run_encoder_oracle(oracle, path, family, width, height, raw, env=environment)
            values["embeddings"] = embeddings.reshape(1, 1, -1, width_out)
            fixtures.update({f"{step}.{k}": np.ascontiguousarray(v) for k, v in values.items()})
        return fixtures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--ops-oracle", type=Path, help="Optional mmproj_ops_oracle executable")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy")
    parser.add_argument("--families", nargs="+", default=[*SINGLE_GRID, *TWO_GRID])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.ops_oracle:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([str(args.ops_oracle.resolve()), directory], check=True)
            for name, npy in OPS_EXPECTATIONS.items():
                np.save(args.output.parent / f"{npy}.npy", np.fromfile(Path(directory) / f"{name}.bin", np.float32))
    for family in args.families:
        make = two_grid_fixture if family in TWO_GRID else single_grid_fixture
        save_npz(args.output / f"{family}.npz", make(args.oracle, family))


if __name__ == "__main__":
    main()

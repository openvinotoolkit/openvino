# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU oracle fixtures for GGUFMMProjDynamicAccuracy: each family runs on two input sizes.

Use PYTHONPATH=<llama.cpp>/gguf-py and --oracle <mmproj_oracle>; see
test_data/mmproj_accuracy/README.md for the reference revisions.
--ops-oracle <mmproj_ops_oracle> also regenerates the standalone op expectations.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np

from mmproj_fixtures import (TensorWriter, finish, gemma4a_positions, merge_window_order, muse_glimmer_indices,
                             run_oracle, save_npz, split_variants)

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
    tensor("v.position_embd.weight", (4, width))
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


def write_model(path, family):
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


def inputs(family, width, height):
    if family == "muse-glimmer":
        raw = np.random.default_rng(42).uniform(-1, 1, (height, width, 3)).astype(np.float32)
        return raw, {"pixel_values": raw.transpose(2, 0, 1)[None], **muse_glimmer_indices(height // 2, width // 2, 2)}
    family, variants = split_variants(family, *VARIANTS)
    overview = "_overview" in variants
    raw = np.random.default_rng(42).normal(.1, .4, (height, width) if family in {"gemma4ua", "gemma4a"} else (height, width, 3)).astype(np.float32)
    if "_low_contrast" in variants:
        raw[:] = np.array([27, 29, 32], np.float32) / 255
    if family == "gemma4ua":
        return raw, {"waveform_frames": raw.T.reshape(1, 1, width, height)}
    if family == "gemma4a":
        return raw, {"features": raw.reshape(1, 1, height, width), **gemma4a_positions(width, 16)}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--ops-oracle", type=Path, help="Optional mmproj_ops_oracle executable")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy")
    parser.add_argument("--families", nargs="+", default=[
        "muse-glimmer", "pixtral", "pixtral_merge", "phi4", "gemma4v", "gemma4uv", "gemma4uv_low_contrast", "gemma4ua",
        "gemma4v_one_sided", "minicpmv4_6", "gemma4a", "deepseekocr", "deepseekocr2", "deepseekocr_resize",
        "deepseekocr_overview", "deepseekocr2_overview",
    ])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.ops_oracle:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([str(args.ops_oracle.resolve()), directory], check=True)
            for name, npy in OPS_EXPECTATIONS.items():
                np.save(args.output.parent / f"{npy}.npy", np.fromfile(Path(directory) / f"{name}.bin", np.float32))
    for family in args.families:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            out = write_model(model, family)
            fixtures = {"model": np.frombuffer(model.read_bytes(), np.uint8)}
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
            modality = "audio" if family in {"gemma4ua", "gemma4a"} else "vision"
            environment = dict(os.environ)
            if family.endswith("_overview"):
                environment["GGUF_ORACLE_OVERVIEW"] = "1"
            for step, (width, height) in enumerate(sizes):
                raw, values = inputs(family, width, height)
                values["embeddings"] = run_oracle(args.oracle, model, modality, width, height, raw,
                                                  environment).reshape(1, 1, -1, out)
                fixtures.update({f"{step}.{k}": np.ascontiguousarray(v) for k, v in values.items()})
            save_npz(args.output / f"{family}.npz", fixtures)


if __name__ == "__main__":
    main()

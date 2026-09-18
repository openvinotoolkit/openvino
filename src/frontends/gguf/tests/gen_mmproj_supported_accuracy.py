# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CPU oracle fixtures for mmproj families paired with registered language backbones.

Reference: llama.cpp 16fb7d9d326a3fe69a331ce5fbe7a679a1a281bb.
Use PYTHONPATH=<llama.cpp>/gguf-py and --oracle <mmproj_oracle>.
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np


def write_model(path, family):
    one_sided = family.endswith("_one_sided")
    family = family.removesuffix("_resize").removesuffix("_overview").removesuffix("_one_sided")
    audio = family in {"gemma4ua", "gemma4a"}
    width, heads, hidden, output = 16, 2, 24, 12
    prefix, modality = ("a.", "audio") if audio else ("v.", "vision")
    w = gguf.GGUFWriter(path, "clip")
    rng = np.random.default_rng(2718)

    def tensor(name, shape, norm=False):
        w.add_tensor(name, (rng.normal(0, .08, shape) + (1 if norm else 0)).astype(np.float32))

    def linear(name, inp, out, bias=True):
        tensor(name + ".weight", (out, inp))
        if bias:
            tensor(name + ".bias", (out,))

    def norm(name, size, bias=True):
        tensor(name + ".weight", (size,), True)
        if bias:
            tensor(name + ".bias", (size,))

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
        w.add_uint32(key + "patch_size", 2)
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
            linear("v.patch_embd", 3 * 4 * 4, width)
            norm("v.patch_norm.1", 48)
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
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return output


def inputs(family, width, height):
    overview = family.endswith("_overview")
    family = family.removesuffix("_resize").removesuffix("_overview").removesuffix("_one_sided")
    raw = np.random.default_rng(42).normal(.1, .4, (height, width) if family in {"gemma4ua", "gemma4a"} else (height, width, 3)).astype(np.float32)
    if family == "gemma4ua":
        return raw, {"waveform_frames": raw.T.reshape(1, 1, width, height)}
    if family == "gemma4a":
        n = (width + 3) // 4
        q, k = np.indices((n, n))
        distance = q - k
        timescale = np.exp(-np.arange(8, dtype=np.float32) * (np.log(np.float32(10000)) / 7))
        theta = np.arange(12, -1, -1, dtype=np.float32)[:, None] * timescale[None]
        return raw, {"features": raw.reshape(1, 1, height, width),
                     "position_embeddings": np.concatenate([np.sin(theta), np.cos(theta)], axis=1)[None, None],
                     "attention_mask": np.where((distance >= 0) & (distance < 12), 0, -1e9).astype(np.float32)[None, None],
                     "relative_indices": np.clip(12 - distance, 0, 12).astype(np.int32)[None, None]}
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
    patch = 4 if family == "gemma4uv" else 2
    h, w = height // patch, width // patch
    rows, cols = np.indices((h, w))
    if family.startswith(("pixtral", "gemma4")):
        result.update(position_x=cols.astype(np.int32).reshape(1, 1, 1, -1),
                      position_y=rows.astype(np.int32).reshape(1, 1, 1, -1))
    if family == "minicpmv4_6":
        result["position_ids"] = (70 * (70 * rows // h) + 70 * cols // w).astype(np.int32).reshape(1, 1, 1, -1)
        order = [(y + dy) * w + x + dx for y in range(0, h, 2) for x in range(0, w, 2)
                 for dy in range(2) for dx in range(2)]
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
    parser.add_argument("--geometry-oracle", type=Path, help="Optional mmproj_ops_oracle executable")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy")
    parser.add_argument("--families", nargs="+", default=[
        "pixtral", "pixtral_merge", "phi4", "gemma4v", "gemma4uv", "gemma4ua",
        "gemma4v_one_sided", "minicpmv4_6", "gemma4a", "deepseekocr", "deepseekocr2", "deepseekocr_resize",
        "deepseekocr_overview", "deepseekocr2_overview",
    ])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.geometry_oracle:
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([str(args.geometry_oracle.resolve()), directory], check=True)
            for name in ("window_input", "windows", "restored", "relative_input", "relative"):
                values = np.fromfile(Path(directory) / f"{name}.bin", np.float32)
                np.save(args.output.parent / f"mmproj_{name}.npy", values)
    for family in args.families:
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            model = d / "model.gguf"
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
            }.get(family, [(16, 8), (8, 24)])
            modality = "audio" if family in {"gemma4ua", "gemma4a"} else "vision"
            environment = dict(os.environ)
            if family.endswith("_overview"):
                environment["GGUF_ORACLE_OVERVIEW"] = "1"
            for step, (width, height) in enumerate(sizes):
                raw, values = inputs(family, width, height)
                raw.tofile(d / "input.bin")
                subprocess.run([str(args.oracle.resolve()), str(model), modality, str(width), str(height),
                                str(d / "input.bin"), str(d / "output.bin")], check=True, env=environment)
                values["embeddings"] = np.fromfile(d / "output.bin", np.float32).reshape(1, 1, -1, out)
                fixtures.update({f"{step}.{k}": np.ascontiguousarray(v) for k, v in values.items()})
            np.savez_compressed(args.output / f"{family}.npz", **fixtures)


if __name__ == "__main__":
    main()

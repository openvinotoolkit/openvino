# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Nonzero LFM3 speech fixtures from llama.cpp CPU at 604cc430239c76ff636c443e3400bf552b750421."""
import argparse
from pathlib import Path
import subprocess

import gguf
import numpy as np

from mmproj_fixtures import TensorWriter, finish, save_npz


class SpeechTensorWriter(TensorWriter):
    def tensor(self, name, shape, norm=False):
        count = int(np.prod(shape))
        if name.startswith("a.gen.wav.") and count > 8192:
            pattern = self.rng.normal(0, self.std, 4093).astype(np.float32)
            self.writer.add_tensor(name, np.resize(pattern, count).reshape(shape))
        else:
            super().tensor(name, shape, norm)


def make_model(path):
    writer = gguf.GGUFWriter(path, "clip")
    writer.add_bool("clip.has_gen_audio_encoder", True)
    writer.add_string("clip.gen.audio.projector_type", "lfm3a_gen")
    for name, value in [("embedding_length", 8), ("feed_forward_length", 12), ("block_count", 3),
                        ("projection_dim", 6), ("attention.head_count", 4), ("attention.head_count_kv", 2)]:
        writer.add_uint32("clip.gen.audio." + name, value)
    writer.add_float32("clip.gen.audio.attention.layer_norm_epsilon", 1e-5)
    tensors = SpeechTensorWriter(writer, np.random.default_rng(7103))
    tensors.tensor("a.gen.code.embd.weight", (12, 2017, 8))
    tensors.tensor("a.gen.code.head_norm.weight", (12, 8), True)
    tensors.tensor("a.gen.code.out_embd.weight", (12 * 2017, 6))
    tensors.tensor("a.gen.code.mask_embd", (8,))
    tensors.norm("a.gen.code.output_norm", 8, False)
    tensors.linear("a.gen.code.proj_in", 6, 12 * 8)
    levels = np.array([8, 7, 6, 6])
    basis = np.array([1, 8, 56, 336])
    codebook = ((np.arange(2016)[:, None] // basis) % levels - levels // 2) / (levels // 2)
    writer.add_tensor("a.gen.wav.quant.first.codebook.weight", codebook.astype(np.float32))
    tensors.linear("a.gen.wav.quant.first.out_proj", 48, 6)
    tensors.linear("a.gen.wav.tfm.in_proj", 6, 512)
    tensors.linear("a.gen.wav.tfm.out_proj", 512, 1282)
    tensors.norm("a.gen.wav.tfm.output_norm", 512, False)
    for base, count in [("a.gen.code", 3), ("a.gen.wav.tfm", 8)]:
        for layer in range(count):
            prefix = f"{base}.blk.{layer}."
            channels = 512 if base.endswith("tfm") else 8
            for name in ["ln1", "ln2"]:
                tensors.norm(prefix + name, channels, False)
            if base.endswith("tfm"):
                for name in ["attn_post_norm", "ffn_post_norm"]:
                    tensors.norm(prefix + name, channels, False)
            if base.endswith("tfm") and layer % 2 == 0:
                tensors.linear(prefix + "conv_pw1", channels, 3 * channels, False)
                tensors.tensor(prefix + "conv_dw.weight", (channels, 4))
                tensors.linear(prefix + "conv_pw2", channels, channels, False)
            else:
                for name, inp, out in [("attn_q", channels, 8), ("attn_k", channels, 4),
                                       ("attn_v", channels, 4), ("attn_out", 8, channels),
                                       ("attn_gate", channels, 4)]:
                    tensors.linear(prefix + name, inp, out, False)
                for name in ["attn_q_norm", "attn_k_norm"]:
                    tensors.norm(prefix + name, 2, False)
            for name, inp, out in [("ffn_gate", channels, 12), ("ffn_up", channels, 12), ("ffn_down", 12, channels)]:
                tensors.linear(prefix + name, inp, out, False)
    finish(writer)


def generate(oracle, output):
    directory = Path(oracle).resolve().parent / "speech-fixture-work"
    directory.mkdir(exist_ok=True)
    model = directory / "model.gguf"
    make_model(model)
    hidden = np.random.default_rng(43).normal(size=(9, 6)).astype(np.float32)
    codes = np.random.default_rng(44).integers(0, 2016, size=(9, 12), dtype=np.int32)
    codes[0, 0], codes[0, 1] = 0, 2015
    depth_codes = codes.copy()
    depth_codes[0, 2], depth_codes[8, 11] = 2016, 2016
    hidden.tofile(directory / "hidden.f32")
    codes.tofile(directory / "codes.i32")
    depth_codes.tofile(directory / "depth_codes.i32")
    reference = directory / "reference"
    subprocess.run([str(Path(oracle).resolve()), str(model), str(directory), str(reference)], check=True)
    arrays = {"model": np.frombuffer(model.read_bytes(), np.uint8), "hidden": hidden, "codes": codes,
              "depth_codes": depth_codes}
    for path in sorted(reference.glob("*.f32")):
        arrays[path.stem] = np.fromfile(path, np.float32)
    save_npz(output, arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data/mmproj_accuracy/lfm3_speech.npz")
    args = parser.parse_args()
    generate(args.oracle, args.output)

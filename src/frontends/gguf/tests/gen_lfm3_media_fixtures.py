# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate LFM3 media expectations with pinned libmtmd CPU, without OpenVINO."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import gguf
import numpy as np

from mmproj_fixtures import TensorWriter, finish, save_npz

REVISION = "604cc430239c76ff636c443e3400bf552b750421"


def write_model(path, modality, projection_width=12):
    writer = gguf.GGUFWriter(path, "clip")
    writer.add_bool(f"clip.has_{modality}_encoder", True)
    writer.add_string(f"clip.{modality}.projector_type", "lfm3vl" if modality == "vision" else "lfm3a")
    writer.add_bool("clip.use_silu", True)
    key = f"clip.{modality}."
    for name, value in {"embedding_length": 16, "feed_forward_length": 24, "block_count": 2,
                        "projection_dim": projection_width, "attention.head_count": 4, "attention.head_count_kv": 2}.items():
        writer.add_uint32(key + name, value)
    writer.add_float32(key + "attention.layer_norm_epsilon", 1e-5)
    tensor = TensorWriter(writer, np.random.default_rng(20261028))
    prefix = "v." if modality == "vision" else "a."
    for layer in range(2):
        p = f"{prefix}blk.{layer}."
        tensor.norm(p + "ln1", 16, False)
        tensor.norm(p + "ln2", 16, False)
        if layer == 0:
            if modality == "vision":
                tensor.linear(p + "conv2d_in_proj", 16, 48, False)
                tensor.tensor(p + "conv2d_spatial.weight", (16, 1, 3, 3))
                tensor.linear(p + "conv2d_out_proj", 16, 16, False)
            else:
                tensor.linear(p + "conv_pw1", 16, 48, False)
                tensor.tensor(p + "conv_dw.weight", (16, 3))
                tensor.linear(p + "conv_pw2", 16, 16, False)
        else:
            tensor.linear(p + "attn_q", 16, 16, False)
            tensor.linear(p + "attn_k", 16, 8, False)
            tensor.linear(p + "attn_v", 16, 8, False)
            tensor.linear(p + "attn_out", 16, 16, False)
            tensor.norm(p + "attn_q_norm", 4, False)
            tensor.norm(p + "attn_k_norm", 4, False)
        tensor.linear(p + "ffn_gate", 16, 24, False)
        tensor.linear(p + "ffn_up", 16, 24, False)
        tensor.linear(p + "ffn_down", 24, 16, False)
    if modality == "vision":
        writer.add_uint32(key + "image_size", 32)
        writer.add_uint32(key + "patch_size", 16)
        writer.add_uint32(key + "spatial_merge_size", 2)
        writer.add_uint32(key + "image_min_pixels", 1024)
        writer.add_uint32(key + "image_max_pixels", 65536)
        writer.add_string(key + "image_resize_algo", "bicubic")
        writer.add_array(key + "image_mean", [0.5, 0.5, 0.5])
        writer.add_array(key + "image_std", [0.5, 0.5, 0.5])
        tensor.tensor("v.patch_embd.weight", (16, 3, 16, 16))
        tensor.tensor("v.patch_embd.weight.1", (16, 3, 16, 16))
        tensor.norm("v.embedding_norm", 16, False)
        tensor.norm("mm.input_norm", 16)
        tensor.linear("mm.0", 64, 24)
        tensor.linear("mm.2", 24, projection_width)
    else:
        writer.add_uint32(key + "num_mel_bins", 128)
        writer.add_uint32(key + "window_size", 4)
        for layer, shape in {0: (2, 1, 3, 3), 2: (2, 1, 3, 3), 3: (2, 2, 1, 1),
                             5: (2, 1, 3, 3), 6: (2, 2, 1, 1)}.items():
            tensor.tensor(f"a.conv1d.{layer}.weight", shape)
            tensor.tensor(f"a.conv1d.{layer}.bias", (2, 1, 1))
        tensor.linear("a.pre_encode.out", 32, 16)
        tensor.norm("a.post_ln", 16, False)
        tensor.norm("mm.a.mlp.0", 16)
        tensor.linear("mm.a.mlp.1", 16, 24)
        tensor.linear("mm.a.mlp.3", 24, projection_width)
    finish(writer)


def generate(oracle, directory, modality, streaming, projection_width=12):
    arrays = {}
    rng = np.random.default_rng(1731)
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        model = scratch / "model.gguf"
        write_model(model, modality, projection_width)
        arrays["model"] = np.frombuffer(model.read_bytes(), np.uint8)
        cases = [(64, 32), (96, 64)] if modality == "vision" else [(24, 128), (17, 128)]
        arguments = [str(oracle), str(model), modality, str(scratch / "oracle")]
        for i, (width, height) in enumerate(cases):
            first = rng.normal(0, 0.4, (height, width, 3) if modality == "vision" else (height, width)).astype(np.float32)
            first.tofile(scratch / f"{i}.f32")
            second_name = "-"
            if modality == "vision":
                second = rng.normal(0, 0.4, first.shape).astype(np.float32)
                second.tofile(scratch / f"{i}.second.f32")
                second_name = str(scratch / f"{i}.second.f32")
                arrays[f"{i}.pixel_values"] = first.transpose(2, 0, 1)[None].copy()
                arrays[f"{i}.pixel_values_1"] = second.transpose(2, 0, 1)[None].copy()
                y, x = np.meshgrid(np.arange(height // 16), np.arange(width // 16), indexing="ij")
                arrays[f"{i}.positions"] = np.stack((y, x, np.zeros_like(y), np.zeros_like(y))).astype(np.int32).reshape(1, 1, 1, -1)
            else:
                arrays[f"{i}.features"] = first.reshape(1, 1, height, width)
            arguments.extend((str(width), str(height), str(scratch / f"{i}.f32"), second_name))
        if modality == "vision" or streaming:
            subprocess.run(arguments, check=True)
        else:
            for i in range(2):
                call = arguments[:4] + arguments[4 + 4 * i:8 + 4 * i]
                call[3] = str(scratch / f"cold.{i}")
                subprocess.run(call, check=True)
                (scratch / "oracle" / str(i)).mkdir(parents=True)
                for file in (scratch / f"cold.{i}" / "0").glob("*.f32"):
                    (scratch / "oracle" / str(i) / file.name).write_bytes(file.read_bytes())
        for i in range(2):
            output = scratch / "oracle" / str(i)
            arrays[f"{i}.embeddings"] = np.fromfile(output / "embeddings.f32", np.float32).reshape(1, -1, projection_width)
            for file in output.glob("state_out_*.f32"):
                arrays[f"{i}.state.{file.stem.removeprefix('state_out_')}"] = np.fromfile(file, np.float32)
    label = modality + ("_stream" if streaming else "")
    target = directory / f"lfm3_media_{label}.npz"
    save_npz(target, arrays)
    return {"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "cases": len(cases)}



def combine_omni(oracle, output, speech_fixture):
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        vision_info = generate(oracle, scratch, "vision", False, 6)
        audio_info = generate(oracle, scratch, "audio", False, 6)
        vision = np.load(scratch / vision_info["file"])
        audio = np.load(scratch / audio_info["file"])
        speech = np.load(speech_fixture)
        writer = gguf.GGUFWriter(scratch / "omni.gguf", "clip")
        metadata = {}
        names = set()
        for i, arrays in enumerate((vision, audio, speech)):
            path = scratch / f"component.{i}.gguf"
            path.write_bytes(arrays["model"].tobytes())
            reader = gguf.GGUFReader(path)
            for name, field in reader.fields.items():
                if name.startswith("GGUF.") or name == "general.architecture":
                    continue
                value = field.contents()
                if name in metadata:
                    if metadata[name] != value:
                        raise ValueError(f"Conflicting component metadata: {name}")
                    continue
                metadata[name] = value
                writer.add_key_value(name, value, field.types[0],
                                     field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
            for tensor in reader.tensors:
                if tensor.name in names:
                    raise ValueError(f"Conflicting component tensor: {tensor.name}")
                names.add(tensor.name)
                writer.add_tensor(tensor.name, tensor.data.copy(), raw_dtype=tensor.tensor_type)
        finish(writer)
        arrays = {"model": np.frombuffer((scratch / "omni.gguf").read_bytes(), np.uint8)}
        for name in ("pixel_values", "pixel_values_1", "positions"):
            arrays["vision." + name] = vision["0." + name]
        arrays["audio.features"] = audio["0.features"]
        arrays["audio.depth.hidden_states"] = speech["hidden"].reshape(1, 1, 9, 6)
        arrays["audio.depth.codes"] = np.zeros((1, 1, 9, 12), np.int32)
        arrays["audio.depth.mask"] = np.zeros((1, 1, 9, 12), np.float32)
        arrays["audio.feedback.codes"] = speech["depth_codes"].reshape(1, 1, 9, 12)
        arrays["audio.codec.codes"] = speech["codes"][:1].reshape(1, 1, 1, 12)
        arrays["audio.codec.positions"] = np.arange(6, dtype=np.int32).reshape(1, 1, 1, 6)
        mask = np.full((6, 35), -np.inf, np.float32)
        for query in range(6):
            for key in range(query, query + 30):
                if key >= 29:
                    mask[query, key] = 0
        arrays["audio.codec.attention_mask"] = mask.reshape(1, 1, 6, 35)
        arrays["expected.vision.embeddings"] = vision["0.embeddings"]
        arrays["expected.audio.embeddings"] = audio["0.embeddings"]
        arrays["expected.audio.depth_logits"] = np.concatenate([speech[f"depth_0_{i}"] for i in range(9)])
        arrays["expected.audio.feedback"] = np.concatenate([speech[f"feedback_{i}"] for i in range(9)])
        arrays["expected.audio.spectrogram"] = speech["spectra_0"]
        for name in audio.files:
            if name.startswith("0.state."):
                arrays["state.audio.state." + name.removeprefix("0.state.")] = audio[name]
        for name in speech.files:
            if name.startswith("state_0_gen_state_"):
                arrays["state.audio.codec.state." + name.removeprefix("state_0_gen_state_").replace("_", ".")] = speech[name]
        target = output / "lfm3_media_omni.npz"
        save_npz(target, arrays)
        return {"file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "independent_components": [vision_info, audio_info,
                    {"file": speech_fixture.name, "sha256": hashlib.sha256(speech_fixture.read_bytes()).hexdigest()}],
                "language_width": 6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-src", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--llama-build", type=Path, required=True)
    parser.add_argument("--omni-speech", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "test_data")
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "-C", str(args.llama_src), "rev-parse", "HEAD"], text=True).strip()
    if actual != REVISION:
        raise ValueError(f"LFM3 CPU oracle must use {REVISION}, got {actual}")
    cache = (args.llama_build / "CMakeCache.txt").read_text()
    if "GGML_OPENVINO:BOOL=OFF" not in cache:
        raise ValueError("CPU reference must be built with GGML_OPENVINO=OFF")
    libraries = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((args.llama_build / "bin").glob("lib*.so"))}
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = [generate(args.oracle.resolve(), args.output, "vision", False),
                generate(args.oracle.resolve(), args.output, "audio", False),
                generate(args.oracle.resolve(), args.output, "audio", True)]
    if args.omni_speech:
        fixtures.append(combine_omni(args.oracle.resolve(), args.output, args.omni_speech))
    manifest = {"llama_cpp": REVISION, "backend": "CPU", "flash_attention": False,
                "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
                "oracle_source_sha256": hashlib.sha256((Path(__file__).parent / "lfm3_media_oracle.cpp").read_bytes()).hexdigest(),
                "libraries": libraries, "fixtures": fixtures}
    (args.output / "lfm3_media_provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

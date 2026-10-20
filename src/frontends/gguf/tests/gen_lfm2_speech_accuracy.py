# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Generate a compact depthformer fixture using the pinned LiquidAI PyTorch reference."""
import argparse
import hashlib
import tempfile
from pathlib import Path
import subprocess
import types

import gguf
import numpy as np
import torch
from transformers import Lfm2Config, Lfm2Model
from gen_arch_accuracy import write_lfm2_model

from mmproj_fixtures import finish, save_npz

REVISION = "19e65845923a7f136442c95137884ec61eb386aa"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquid-audio", required=True, type=Path)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.liquid_audio), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected {REVISION}, got {revision}")
    source = args.liquid_audio / "src/liquid_audio/model/transformer.py"
    code = source.read_text().replace("type CacheType =", "CacheType =").replace(
        "def wrap_activation_checkpoint[T: nn.Module]", "def wrap_activation_checkpoint")
    module = types.ModuleType("liquid_reference")
    exec(compile("from __future__ import annotations\n" + code, str(source), "exec"), module.__dict__)
    torch.manual_seed(612)
    width, books, vocabulary, language, codec = 16, 8, 2049, 12, 6
    model = module.RawLMBackbone([
        module.StandardBlock(module.MHA(width, num_heads=4, gqa_dim=2), ff_dim=24, multiple_of=8)
        for _ in range(2)], has_embedding=False).eval()
    projection = torch.nn.Linear(language, width * books)
    embeddings = [module.SharedEmbedding(width, vocabulary, tie_embedding=False) for _ in range(books)]
    audio_embedding = torch.nn.Embedding(books * vocabulary, language)
    codec_embedding = torch.nn.Embedding(books * (vocabulary - 1), codec)
    for value in list(model.parameters()) + list(projection.parameters()) + [p for e in embeddings for p in e.parameters()] + list(audio_embedding.parameters()) + list(codec_embedding.parameters()):
        with torch.no_grad():
            if value.ndim == 1 and value.numel() != width * books:
                value.copy_(1 + .08 * torch.randn_like(value))
            else:
                value.copy_(.08 * torch.randn_like(value))
    hidden = torch.randn(1, 1, language) * .3
    depth_codes = torch.randint(0, vocabulary - 1, (1, books), dtype=torch.int32)
    codes = torch.randint(0, vocabulary - 1, (1, 3, books), dtype=torch.int32)
    with torch.no_grad():
        x = projection(hidden).reshape(1, books, width)
        for i in range(1, books):
            x[:, i] += embeddings[i - 1](depth_codes[:, i - 1].long())
        y = model(x)
        logits = torch.stack([embeddings[i].get_logits(y[:, i]) for i in range(books)], 1)
        feedback = audio_embedding(codes.long() + torch.arange(books) * vocabulary).sum(2)
        features = codec_embedding(codes.long() + torch.arange(books) * (vocabulary - 1)).mean(2)
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "vocoder.gguf"
    weights = {"depthformer." + key.replace("operator.bounded_attention.", "operator.attention."): value
               for key, value in model.state_dict().items()}
    weights.update({"depth_linear." + key: value for key, value in projection.state_dict().items()})
    for i, embedding in enumerate(embeddings):
        weights.update({f"depth_embeddings.{i}." + key: value for key, value in embedding.state_dict().items()})
    weights["audio_embedding.embedding.weight"] = audio_embedding.weight
    weights["emb.emb.weight"] = codec_embedding.weight
    weights["istft.window"] = torch.hann_window(1280)
    arrays = {"hidden": hidden.numpy().reshape(1, 1, 1, language),
              "depth_codes": depth_codes.numpy().reshape(1, 1, 1, books),
              "codes": codes.numpy().reshape(1, 1, 3, books),
              "logits": logits.numpy(), "feedback": feedback.numpy(), "codec": features.numpy(),
              "reference_revision": np.frombuffer(REVISION.encode(), np.uint8),
              "reference_sha256": np.frombuffer(hashlib.sha256(source.read_bytes()).hexdigest().encode(), np.uint8)}
    for key, architecture in [
        ("model", "lfm2.audio_vocoder"),
        ("publisher_model", "this model cannot be used as LLM, use it via --model-vocoder in TTS examples"),
    ]:
        writer = gguf.GGUFWriter(path, architecture)
        writer.add_uint32("depthformer_n_layer", 2)
        writer.add_uint32("depthformer_n_embd", width)
        for name, value in weights.items():
            writer.add_tensor(name, value.detach().numpy())
        finish(writer)
        arrays[key] = np.frombuffer(path.read_bytes(), np.uint8)
    save_npz(Path(__file__).parent / "test_data/lfm2_speech.npz", arrays)
    detok_path = Path(temporary.name) / "detokenizer.gguf"
    write_lfm2_model(detok_path, "lfm2", {"tied": True})
    reader = gguf.GGUFReader(detok_path)
    weights = {t.name: torch.from_numpy(t.data.copy().reshape(tuple(reversed(t.shape)))) for t in reader.tensors}
    weights["dense_2.weight"] = torch.randn(1282, 32) * .08
    weights["dense_2.bias"] = torch.randn(1282) * .08
    writer = gguf.GGUFWriter(detok_path, "lfm2")
    for field in reader.fields.values():
        if field.name.startswith("GGUF.") or field.name == "general.architecture":
            continue
        value = field.contents()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        writer.add_key_value(field.name, value, field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_sliding_window(30)
    for name, value in weights.items():
        writer.add_tensor(name, value.numpy())
    finish(writer)
    config = Lfm2Config(hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, vocab_size=32,
                       layer_types=["conv", "full_attention", "conv", "full_attention"],
                       conv_L_cache=3, norm_eps=1e-5, rope_theta=1000000.,
                       block_auto_adjust_ff_dim=False)
    detok = Lfm2Model(config).eval()
    state = {"embed_tokens.weight": weights["token_embd.weight"],
             "embedding_norm.weight": weights["token_embd_norm.weight"]}
    for i in range(4):
        p, target = f"blk.{i}.", f"layers.{i}."
        for a, b in [("attn_norm.weight", "operator_norm.weight"), ("ffn_norm.weight", "ffn_norm.weight"),
                     ("ffn_gate.weight", "feed_forward.w1.weight"), ("ffn_down.weight", "feed_forward.w2.weight"),
                     ("ffn_up.weight", "feed_forward.w3.weight")]:
            state[target + b] = weights[p + a]
        if p + "shortconv.conv.weight" in weights:
            for name in ["in_proj", "out_proj", "conv"]:
                value = weights[p + "shortconv." + name + ".weight"]
                state[target + "conv." + name + ".weight"] = value.unsqueeze(1) if name == "conv" else value
        else:
            for a, b in [("attn_q", "q_proj"), ("attn_k", "k_proj"), ("attn_v", "v_proj"),
                         ("attn_output", "out_proj"), ("attn_q_norm", "q_layernorm"), ("attn_k_norm", "k_layernorm")]:
                state[target + "self_attn." + b + ".weight"] = weights[p + a + ".weight"]
    detok.load_state_dict(state)
    embeds = torch.randn(1, 37, 32) * .3
    positions = torch.arange(37)
    mask = ((positions - positions[:, None]) <= 0) & ((positions - positions[:, None]) > -30)
    with torch.no_grad():
        hidden = detok(inputs_embeds=embeds, attention_mask=mask[None, None], use_cache=False).last_hidden_state
        spectral = torch.nn.functional.linear(hidden, weights["dense_2.weight"], weights["dense_2.bias"])
    save_npz(Path(__file__).parent / "test_data/lfm2_detokenizer.npz",
             {"model": np.frombuffer(detok_path.read_bytes(), np.uint8),
              "embeds": embeds.numpy(), "spectral": spectral.numpy()})


if __name__ == "__main__":
    main()

# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import torch
from transformers.activations import ACT2FN


@dataclass(frozen=True)
class PrimitiveCase:
    name: str
    family: str


class FunctionModule(torch.nn.Module):
    def __init__(self, function, **kwargs):
        super().__init__()
        self.function = function
        self.kwargs = kwargs

    def forward(self, *inputs):
        return self.function(*inputs, **self.kwargs)


class Attention(torch.nn.Module):
    def __init__(self, backend, groups, causal, masked):
        super().__init__()
        self.num_key_value_groups = groups
        self.is_causal = causal
        self.masked = masked
        if backend == "sdpa":
            from transformers.integrations.sdpa_attention import sdpa_attention_forward
            self.function = sdpa_attention_forward
        else:
            from transformers.models.llama.modeling_llama import eager_attention_forward
            self.function = eager_attention_forward

    def forward(self, query, key, value, mask):
        return self.function(self, query, key, value, mask if self.masked else None, scaling=0.5)[0]


class CacheOperations(torch.nn.Module):
    def __init__(self, kind, operation):
        super().__init__()
        self.kind = kind
        self.operation = operation

    def forward(self, past_key, past_value, key, value, beam_indices):
        from transformers import cache_utils

        if self.kind == "dynamic":
            cache = cache_utils.DynamicLayer()
        elif self.kind == "sliding":
            cache = cache_utils.DynamicSlidingWindowLayer(sliding_window=4)
        elif self.kind == "static":
            cache = cache_utils.StaticLayer(max_cache_len=8)
        else:
            cache = cache_utils.StaticSlidingWindowLayer(max_cache_len=8, sliding_window=4)
        if self.kind in {"static", "static_sliding"}:
            cache.cumulative_length = past_key.new_zeros((), dtype=torch.long)
        cache.update(past_key, past_value)
        full_key, full_value = cache.update(key, value)
        if self.operation == "reorder":
            cache.reorder_cache(beam_indices)
        elif self.operation == "repeat":
            cache.batch_repeat_interleave(2)
        elif self.operation == "crop":
            cache.crop(-1)
        return full_key.clone(), full_value.clone(), cache.keys.clone(), cache.values.clone()


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, rope_type):
        super().__init__()
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        parameters = {"rope_type": rope_type, "rope_theta": 10000.0}
        if rope_type not in {"default", "proportional"}:
            parameters["factor"] = 2.0
        if rope_type == "llama3":
            parameters.update(low_freq_factor=1.0, high_freq_factor=4.0, original_max_position_embeddings=8)
        elif rope_type == "proportional":
            parameters.update(partial_rotary_factor=0.5)
        elif rope_type == "longrope":
            parameters.update(short_factor=[1.0] * 4, long_factor=[2.0] * 4)
        config = LlamaConfig(hidden_size=32, num_attention_heads=4, max_position_embeddings=16,
                             rope_parameters=parameters)
        self.rope = LlamaRotaryEmbedding(config)

    def forward(self, query, key, positions):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        cos, sin = self.rope(query, positions)
        rotary_dim = cos.shape[-1]
        rotated_query, rotated_key = apply_rotary_pos_emb(query[..., :rotary_dim], key[..., :rotary_dim], cos, sin)
        return (torch.cat((rotated_query, query[..., rotary_dim:]), dim=-1),
                torch.cat((rotated_key, key[..., rotary_dim:]), dim=-1))


class VisionEmbedding(torch.nn.Module):
    def __init__(self, interpolate):
        super().__init__()
        from transformers import ViTConfig
        from transformers.models.vit.modeling_vit import ViTEmbeddings

        self.embeddings = ViTEmbeddings(ViTConfig(image_size=16, patch_size=4, hidden_size=24), use_mask_token=True)
        self.interpolate = interpolate

    def forward(self, pixels, mask):
        return self.embeddings(pixels, bool_masked_pos=mask, interpolate_pos_encoding=self.interpolate)


class SinePositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers.models.detr.modeling_detr import DetrSinePositionEmbedding

        self.embedding = DetrSinePositionEmbedding(num_position_features=8, normalize=True)

    def forward(self, pixels, mask):
        return self.embedding(pixels.shape, pixels.device, pixels.dtype, mask)


def primitive_cases():
    cases = [PrimitiveCase(name, "activation") for name in sorted(ACT2FN)]
    cases += [PrimitiveCase(name, "utility")
              for name in ("conv1d", "chunking", "meshgrid", "rms_norm", "gated_mlp", "relative_position")]
    cases += [PrimitiveCase(f"{backend}-{pattern}-{phase}", "mask")
              for backend in ("eager", "sdpa")
              for pattern in ("causal", "bidirectional", "sliding", "chunked", "packed")
              for phase in ("prefill", "decode")]
    cases += [PrimitiveCase(f"{backend}-{groups}-{phase}-{masked}", "attention")
              for backend in ("eager", "sdpa") for groups in (1, 2)
              for phase in ("prefill", "decode", "cross") for masked in (False, True)]
    cases += [PrimitiveCase(f"{kind}-{operation}-{phase}", "cache")
              for kind, operations in (("dynamic", ("update", "reorder", "repeat", "crop")),
                                       ("sliding", ("update", "reorder")),
                                       ("static", ("update", "reorder")),
                                       ("static_sliding", ("update", "reorder")))
              for operation in operations for phase in ("prefill", "decode")]
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    cases += [PrimitiveCase(f"{kind}-{phase}", "rope")
              for kind in ("default", *sorted(ROPE_INIT_FUNCTIONS)) for phase in ("short", "long")]
    cases += [PrimitiveCase(name, "vision")
              for name in ("patch_embedding", "patch_interpolation", "sine_position", "window_partition")]
    cases += [PrimitiveCase(name, "audio")
              for name in ("group_norm", "layer_norm", "positional_conv")]
    cases += [PrimitiveCase(name, "time_series") for name in ("patchify", "std_scaler", "mean_scaler")]
    cases += [PrimitiveCase(name, "generation") for name in (
        "TemperatureLogitsWarper", "TopKLogitsWarper", "TopPLogitsWarper", "MinPLogitsWarper",
        "TypicalLogitsWarper", "EpsilonLogitsWarper", "EtaLogitsWarper", "RepetitionPenaltyLogitsProcessor",
        "MinLengthLogitsProcessor", "ForcedBOSTokenLogitsProcessor", "ForcedEOSTokenLogitsProcessor",
        "SuppressTokensLogitsProcessor", "InfNanRemoveLogitsProcessor", "LogitNormalization")]
    from model_primitives import model_cases
    cases += [PrimitiveCase(f"{model}/{symbol}", "model") for model, symbol in model_cases()]
    from component_adapters import component_cases
    cases += [PrimitiveCase(name, "component") for name in component_cases()]
    return cases


def make_primitive(case):
    torch.manual_seed(0)
    name, family = case.name, case.family
    if family == "component":
        from component_adapters import make_component
        return make_component(name)
    if family == "model":
        from model_primitives import make_model_primitive
        return make_model_primitive(name, FunctionModule)
    if family == "activation":
        return ACT2FN[name].eval(), (torch.linspace(-12, 12, 192).reshape(2, 8, 12),)
    if family == "utility":
        from transformers import pytorch_utils
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP
        from transformers import LlamaConfig

        inputs = (torch.randn(2, 8, 32),)
        if name == "conv1d":
            model = pytorch_utils.Conv1D(24, 32)
        elif name == "chunking":
            model = FunctionModule(partial(pytorch_utils.apply_chunking_to_forward, ACT2FN["gelu_new"].forward, 2, 1))
        elif name == "meshgrid":
            model = FunctionModule(pytorch_utils.meshgrid, indexing="ij")
            inputs = (torch.arange(4), torch.arange(6))
        elif name == "rms_norm":
            model = LlamaRMSNorm(32)
        elif name == "gated_mlp":
            model = LlamaMLP(LlamaConfig(hidden_size=32, intermediate_size=48, hidden_act="silu"))
        else:
            from transformers.models.t5.modeling_t5 import T5Attention
            model = FunctionModule(T5Attention._relative_position_bucket, bidirectional=True)
            inputs = (torch.arange(-128, 128).reshape(16, 16),)
        return model.eval(), inputs
    if family == "mask":
        from transformers import masking_utils as masks

        backend, pattern, phase = name.split("-")
        q_length, offset = (4, 0) if phase == "prefill" else (1, 3)
        functions = {"causal": masks.causal_mask_function, "bidirectional": masks.bidirectional_mask_function,
                     "sliding": masks.sliding_window_causal_mask_function(3),
                     "chunked": masks.chunked_causal_mask_function(2, torch.zeros(2, dtype=torch.long)),
                     "packed": masks.packed_sequence_mask_function(torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]]))}
        model = FunctionModule(getattr(masks, f"{backend}_mask"), batch_size=2, q_length=q_length,
                               kv_length=4, q_offset=offset, mask_function=functions[pattern],
                               allow_is_causal_skip=False, allow_is_bidirectional_skip=False)
        # attention_mask is a keyword argument in the library API.
        model = MaskInput(model)
        return model, (torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.bool),)
    if family == "attention":
        backend, groups, phase, masked = name.split("-")
        groups = int(groups)
        q_length = 1 if phase == "decode" else 4
        kv_length = 6 if phase == "cross" else 4
        causal = phase != "cross"
        mask = torch.zeros(2, 1, q_length, kv_length)
        if causal:
            offset = kv_length - q_length
            allowed = torch.arange(kv_length)[None, :] <= torch.arange(q_length)[:, None] + offset
            mask.masked_fill_(~allowed, float("-inf"))
        mask[..., -1] = float("-inf")
        return Attention(backend, groups, causal, masked == "True").eval(), (
            torch.randn(2, 4, q_length, 8), torch.randn(2, 4 // groups, kv_length, 8),
            torch.randn(2, 4 // groups, kv_length, 8), mask)
    if family == "cache":
        kind, operation, phase = name.split("-")
        past_length = 2 if phase == "prefill" else 5
        length = 3 if phase == "prefill" else 1
        return CacheOperations(kind, operation), (
            torch.randn(2, 2, past_length, 8), torch.randn(2, 2, past_length, 8),
            torch.randn(2, 2, length, 8), torch.randn(2, 2, length, 8), torch.tensor([1, 0]))
    if family == "rope":
        kind, phase = name.split("-")
        offset = 0 if phase == "short" else 32
        return RotaryEmbedding(kind).eval(), (torch.randn(2, 4, 4, 8), torch.randn(2, 2, 4, 8),
                                             torch.arange(offset, offset + 4).expand(2, -1))
    if family == "vision":
        if name in {"patch_embedding", "patch_interpolation"}:
            interpolate = name == "patch_interpolation"
            height = 24 if interpolate else 16
            return VisionEmbedding(interpolate).eval(), (torch.randn(2, 3, height, 16),
                                                        torch.rand(2, height) > 0.5)
        if name == "sine_position":
            return SinePositionEmbedding(), (torch.randn(2, 3, 4, 6), torch.rand(2, 4, 6) > 0.2)
        from transformers.models.swin.modeling_swin import window_partition
        return FunctionModule(window_partition, window_size=4), (torch.randn(2, 8, 8, 12),)
    if family == "audio":
        from transformers import Wav2Vec2Config
        from transformers.models.wav2vec2 import modeling_wav2vec2 as audio

        config = Wav2Vec2Config(hidden_size=16, conv_dim=(8,), conv_stride=(2,), conv_kernel=(3,),
                               num_conv_pos_embeddings=4, num_conv_pos_embedding_groups=4)
        if name == "positional_conv":
            return audio.Wav2Vec2PositionalConvEmbedding(config).eval(), (torch.randn(2, 12, 16),)
        cls = audio.Wav2Vec2GroupNormConvLayer if name == "group_norm" else audio.Wav2Vec2LayerNormConvLayer
        return cls(config, layer_id=0).eval(), (torch.randn(2, 1, 32),)
    if family == "time_series":
        from transformers import PatchTSTConfig
        from transformers.models.patchtst import modeling_patchtst as series

        config = PatchTSTConfig(context_length=16, patch_length=4, patch_stride=3, num_input_channels=3)
        data = torch.randn(2, 16, 3)
        if name == "patchify":
            return series.PatchTSTPatchify(config), (data,)
        cls = series.PatchTSTStdScaler if name == "std_scaler" else series.PatchTSTMeanScaler
        mask = torch.rand(2, 16, 3) > 0.3
        mask[0, :, 0] = False
        return cls(config), (data, mask)
    if family == "generation":
        from transformers.generation import logits_process

        kwargs = {
            "TemperatureLogitsWarper": {"temperature": 0.7}, "TopKLogitsWarper": {"top_k": 5},
            "TopPLogitsWarper": {"top_p": 0.8}, "MinPLogitsWarper": {"min_p": 0.1},
            "TypicalLogitsWarper": {"mass": 0.8}, "EpsilonLogitsWarper": {"epsilon": 0.05},
            "EtaLogitsWarper": {"epsilon": 0.05}, "RepetitionPenaltyLogitsProcessor": {"penalty": 1.2},
            "MinLengthLogitsProcessor": {"min_length": 8, "eos_token_id": [1, 2]},
            "ForcedBOSTokenLogitsProcessor": {"bos_token_id": 1},
            "ForcedEOSTokenLogitsProcessor": {"max_length": 5, "eos_token_id": [1, 2]},
            "SuppressTokensLogitsProcessor": {"suppress_tokens": [0, 3]},
        }
        processor = getattr(logits_process, name)(**kwargs.get(name, {}))
        inputs = torch.tensor([[3, 4, 3, 5], [6, 7, 8, 7]])
        if name == "ForcedBOSTokenLogitsProcessor":
            inputs = inputs[:, :1]
        scores = torch.randn(2, 16)
        if name == "InfNanRemoveLogitsProcessor":
            scores[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
        return FunctionModule(processor), (inputs, scores)
    raise ValueError(f"Unknown primitive {case}")


class MaskInput(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, attention_mask):
        return self.module.function(attention_mask=attention_mask, **self.module.kwargs)

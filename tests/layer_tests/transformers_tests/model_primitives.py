# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import importlib
from pathlib import Path

import torch
import transformers


FUNCTION_RECIPES = {
    "rotate_half": ("llama", "blt", "deepseek_v4", "musicflamingo", "nanochat"),
    "repeat_kv": ("llama",),
    "apply_rotary_pos_emb": ("llama", "bamba", "cohere", "glm", "nemotron", "phi3", "codegen",
                             "dinov3_vit", "deepseek_v4", "diffusion_gemma", "ernie4_5", "ernie4_5_vl_moe",
                             "clvp", "esm", "esmfold2", "flex_olmo", "glm4_moe", "glm4v", "glmasr",
                             "gpt_oss", "helium", "hunyuan_vl", "muse_glimmer_assistant", "pe_audio", "qwen4_exp"),
    "eager_attention_forward": ("llama", "albert", "align", "audio_spectrogram_transformer", "bigbird_pegasus",
                                "blip_2", "decision_transformer", "gemma2", "deepseek_v4", "granite_swa",
                                "llama4", "longt5", "mimo_v2_flash", "modernbert_decoder", "openai_privacy_filter",
                                "aimv2", "evolla", "idefics2", "inkling", "internvl", "vjepa2"),
}
NORM_RECIPES = {
    "llama": "LlamaRMSNorm", "gemma2": "Gemma2RMSNorm", "cohere": "CohereLayerNorm",
    "olmo": "OlmoLayerNorm", "t5": "T5LayerNorm", "deberta": "DebertaLayerNorm",
    "convnext": "ConvNextLayerNorm", "qwen3": "Qwen3RMSNorm", "deepseek_v3": "DeepseekV3RMSNorm",
    "jamba": "JambaRMSNorm", "mamba2": "Mamba2RMSNorm", "phi3": "Phi3RMSNorm", "glm": "GlmRMSNorm",
}


def model_cases():
    return [(model, name) for name, models in FUNCTION_RECIPES.items() for model in models] + list(NORM_RECIPES.items())


def implementation_inventory():
    root = Path(transformers.__file__).parent
    tested = set(model_cases())
    records = []
    for path in sorted((root / "models").rglob("modeling_*.py")):
        if path.name.startswith(("modeling_tf_", "modeling_flax_")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_RECIPES:
                body = node.body
            elif isinstance(node, ast.ClassDef) and node.name.endswith(("RMSNorm", "LayerNorm")):
                body = [method for method in node.body if isinstance(method, ast.FunctionDef)
                        and method.name in {"forward", "_norm"}]
            else:
                continue
            body = [statement for statement in body if not (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str))]
            fingerprint = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()
            records.append({"file": str(path.relative_to(root)), "name": node.name,
                            "implementation_sha256": fingerprint,
                            "direct_recipe": (path.parent.name, node.name) in tested})
    return records


class ModelAttention(torch.nn.Module):
    def __init__(self, function, model):
        super().__init__()
        self.function = function
        self.model = model
        self.num_key_value_groups = 1
        self.head_dim = 8
        self.register_buffer("sinks", torch.linspace(-1, 1, 4))

    def forward(self, query, key, value, mask, bias):
        kwargs = {"scaling": 0.5, "dropout": 0.0}
        if self.model == "gemma2":
            kwargs["softcap"] = 2.0
        if self.model in {"longt5", "inkling"}:
            kwargs["position_bias"] = bias
        return self.function(self, query, key, value, mask, **kwargs)


def make_model_primitive(name, function_module):
    model_name, symbol = name.split("/", 1)
    module = importlib.import_module(f"transformers.models.{model_name}.modeling_{model_name}")
    implementation = getattr(module, symbol)
    x = torch.randn(2, 4, 4, 8)
    if symbol == "rotate_half":
        return function_module(implementation), (x,)
    if symbol == "repeat_kv":
        return function_module(implementation, n_rep=2), (x[:, :2],)
    if symbol == "eager_attention_forward":
        mask = torch.zeros(2, 1, 4, 4)
        mask[..., -1] = float("-inf")
        return ModelAttention(implementation, model_name).eval(), (x, torch.randn_like(x), torch.randn_like(x),
                                                                mask, torch.randn(1, 4, 4, 4) * 0.1)
    if symbol == "apply_rotary_pos_emb":
        cos, sin = torch.randn(2, 4, 8), torch.randn(2, 4, 8)
        if model_name == "clvp":
            return function_module(implementation), (x, torch.randn_like(x), torch.randn_like(x),
                                                     cos[0], sin[0], torch.arange(4).expand(2, -1))
        if model_name == "muse_glimmer_assistant":
            return function_module(implementation), (x[:, :, -1:], torch.randn_like(x), cos, sin)
        if model_name == "codegen":
            return function_module(implementation), (x.transpose(1, 2), sin[..., :4], cos[..., :4])
        if model_name == "dinov3_vit":
            return function_module(implementation), (x, torch.randn_like(x), cos[:, None, :3], sin[:, None, :3])
        if model_name in {"deepseek_v4", "diffusion_gemma"}:
            width = 2 if model_name == "deepseek_v4" else 8
            return function_module(implementation), (x, cos[..., :width], sin[..., :width])
        if model_name in {"bamba", "nemotron", "phi3", "glm4_moe", "glm4v", "glmasr", "qwen4_exp", "gpt_oss"}:
            cos, sin = cos[..., :4], sin[..., :4]
        return function_module(implementation), (x, torch.randn_like(x), cos, sin)
    kwargs = {"data_format": "channels_first"} if model_name == "convnext" else {}
    norm = implementation(8, **kwargs).eval()
    with torch.no_grad():
        if getattr(norm, "weight", None) is not None:
            norm.weight.copy_(torch.linspace(0.5, 1.5, norm.weight.numel()).reshape(norm.weight.shape))
    return norm, (torch.randn(2, 8, 4, 4) if model_name == "convnext" else x,)

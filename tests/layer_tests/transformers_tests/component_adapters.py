# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers

from model_primitives import model_cases


MLP_FIELDS = {"hidden_size", "intermediate_size", "hidden_act", "hidden_activation", "mlp_bias"}
MLP_CALLS = {"super", "super().__init__", "nn.Linear", "torch.nn.Linear"}
SPECIAL_CASES = {
    "llama/LlamaAttention": ("prefill", "single_token"),
    "qwen3/Qwen3Attention": ("prefill", "single_token"),
    "phi3/Phi3MLP": ("default",),
    "deepseek_v2/DeepseekV2TopkRouter": ("greedy", "group_limited_greedy"),
    "bert/BertEmbeddings": ("default",),
    "deepseek_v2/DeepseekV2RotaryEmbedding": tuple(
        f"{kind}-{length}" for kind in ("default", "linear", "yarn") for length in ("short", "long")),
}


def simple_mlp(node):
    if not node.name.endswith("MLP"):
        return False
    methods = {method.name: method for method in node.body if isinstance(method, ast.FunctionDef)}
    init, forward = methods.get("__init__"), methods.get("forward")
    if init is None or forward is None or [arg.arg for arg in init.args.args] != ["self", "config"]:
        return False
    if len(forward.args.args) != 2 or forward.args.vararg or forward.args.kwarg:
        return False
    fields = {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
              and isinstance(item.value, ast.Name) and item.value.id == "config"}
    if not fields or not fields <= MLP_FIELDS:
        return False
    calls = {ast.unparse(item.func) for item in ast.walk(init) if isinstance(item, ast.Call)}
    if not calls <= MLP_CALLS:
        return False
    return not any(isinstance(item, (ast.If, ast.For, ast.While, ast.Try)) for item in ast.walk(forward))


@lru_cache(maxsize=1)
def component_inventory():
    root = Path(transformers.__file__).parent
    explicit = {f"{model}/{symbol}" for model, symbol in model_cases()}
    records = []
    for path in sorted((root / "models").rglob("modeling_*.py")):
        if path.name.startswith(("modeling_tf_", "modeling_flax_")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

        def model_class(node, seen=()):
            for base in node.bases:
                name = ast.unparse(base).split(".")[-1]
                if "PreTrainedModel" in name or name.startswith("GenericFor"):
                    return True
                if name in classes and name not in seen and model_class(classes[name], (*seen, name)):
                    return True
            return False

        def module_class(node, seen=()):
            for base in node.bases:
                name = ast.unparse(base).split(".")[-1]
                if name in {"Module", "Linear", "Embedding", "LayerNorm", "Conv1d", "Conv2d",
                            "GradientCheckpointingLayer"}:
                    return True
                if name in classes and name not in seen and module_class(classes[name], (*seen, name)):
                    return True
            return False

        def record(node, kind, parent=None):
            symbol = f"{parent}.{node.name}" if parent else node.name
            key = f"{path.parent.name}/{symbol}"
            body = [item for item in node.body if not (isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Constant) and isinstance(item.value.value, str))]
            fingerprint = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[])).encode()).hexdigest()
            adapter = "explicit" if key in explicit else "special" if key in SPECIAL_CASES else None
            if kind == "module" and simple_mlp(node) and adapter is None:
                adapter = "linear_mlp"
            excluded = kind == "model"
            records.append({"file": str(path.relative_to(root)), "symbol": symbol, "kind": kind,
                            "implementation_sha256": fingerprint, "adapter": adapter,
                            "status": "excluded" if excluded else "tested" if adapter else "needs_adapter",
                            "reason": "Whole-model orchestration is outside component conversion tests" if excluded
                            else "Direct component construction and inputs" if adapter
                            else "Requires a reviewed constructor/input adapter"})

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [item for item in node.body if isinstance(item, ast.FunctionDef)]
                if model_class(node):
                    record(node, "model")
                elif any(item.name == "forward" for item in methods) or module_class(node):
                    record(node, "module")
                for method in methods:
                    if method.name not in {"__init__", "forward"} and tensor_helper(method):
                        record(method, "method", node.name)
            elif isinstance(node, ast.FunctionDef) and tensor_helper(node):
                record(node, "function")
    return records


def tensor_helper(node):
    text = ast.unparse(node)
    return "torch." in text or "nn." in text or "F." in text or "Tensor" in text or node.name in {
        "rotate_half", "repeat_kv", "apply_rotary_pos_emb", "eager_attention_forward"}


def component_cases():
    cases = []
    for record in component_inventory():
        key = f"{Path(record['file']).parent.name}/{record['symbol']}"
        if record["adapter"] == "linear_mlp":
            cases.append(f"{key}/default")
        elif record["adapter"] == "special":
            cases.extend(f"{key}/{variant}" for variant in SPECIAL_CASES[key])
    return cases


class ComplexComponent(torch.nn.Module):
    def __init__(self, component):
        super().__init__()
        self.component = component

    def forward(self, x, positions):
        return torch.view_as_real(self.component(x, positions))


class AttentionComponent(torch.nn.Module):
    def __init__(self, component):
        super().__init__()
        self.component = component

    def forward(self, hidden, mask, cos, sin):
        return self.component(hidden, position_embeddings=(cos, sin), attention_mask=mask)[0]


def make_component(name):
    model, symbol, variant = name.split("/")
    module = importlib.import_module(f"transformers.models.{model}.modeling_{model}")
    implementation = getattr(module, symbol)
    if symbol.endswith("MLP"):
        config = SimpleNamespace(hidden_size=16, intermediate_size=24, hidden_act="silu",
                                 hidden_activation="gelu_pytorch_tanh", mlp_bias=True)
        return implementation(config), (torch.randn(2, 5, 16),)
    if symbol.endswith("Attention"):
        config_class = getattr(module, f"{symbol.removesuffix('Attention')}Config")
        config = config_class(hidden_size=16, num_attention_heads=2, num_key_value_heads=1,
                              head_dim=8, num_hidden_layers=1, attention_dropout=0.0)
        config._attn_implementation = "eager"
        query_length = 5 if variant == "prefill" else 1
        component = implementation(config, layer_idx=0)
        mask = torch.full((query_length, query_length), float("-inf")).triu(1)
        return AttentionComponent(component), (torch.randn(2, query_length, 16),
                                               mask.expand(2, 1, -1, -1),
                                               torch.randn(2, query_length, 8), torch.randn(2, query_length, 8))
    if symbol == "DeepseekV2TopkRouter":
        config = module.DeepseekV2Config(hidden_size=16, num_attention_heads=2, num_experts=4, num_experts_per_tok=2,
                                        topk_method=variant, n_group=2, topk_group=1)
        router = implementation(config)
        with torch.no_grad():
            router.weight.copy_(torch.randn_like(router.weight) * 0.1)
        return router, (torch.randn(2, 5, 16),)
    if symbol == "DeepseekV2RotaryEmbedding":
        kind, length = variant.split("-")
        fixture = json.loads((Path(__file__).parent / "component_fixtures.json").read_text())["fixtures"][0]["reused"]
        parameters = {"rope_type": kind, "rope_theta": fixture["rope_theta"]}
        if kind != "default":
            parameters["factor"] = fixture["scaling_factor"]
        config = module.DeepseekV2Config(hidden_size=16, num_attention_heads=2, head_dim=8,
                                        max_position_embeddings=16, rope_parameters=parameters)
        input_length = fixture["short_length"] if length == "short" else int(
            config.max_position_embeddings * fixture["long_length_factor"])
        positions = torch.arange(input_length).unsqueeze(0)
        return ComplexComponent(implementation(config)), (torch.randn(1), positions)
    config = module.BertConfig(vocab_size=32, hidden_size=16, max_position_embeddings=16,
                               hidden_dropout_prob=0.0)
    return implementation(config), (torch.randint(0, 32, (2, 5)),)

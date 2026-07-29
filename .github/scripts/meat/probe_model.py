# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Probe-model step for the analyze-and-convert agent.

First step in the analyze-and-convert workflow. Probes a HuggingFace model
to collect its full profile and writes model_profile.json consumed by all
subsequent skills.

Usage:
    python .github/scripts/meat/probe_model.py <model_id>

    MODEL_ID env var is also accepted as a fallback.

Output files (current working directory):
    model_profile.json
"""

import json
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve MODEL_ID
# ---------------------------------------------------------------------------

if len(sys.argv) >= 2:
    MODEL_ID = sys.argv[1]
else:
    MODEL_ID = os.environ.get("MODEL_ID", "")

if not MODEL_ID:
    print(
        "Usage: probe_model.py <model_id>  OR  set MODEL_ID env var",
        file=sys.stderr,
    )
    sys.exit(1)

print(f"[probe_model] Probing model: {MODEL_ID}")

# ---------------------------------------------------------------------------
# Step 1 — Fetch HuggingFace metadata
# ---------------------------------------------------------------------------

try:
    import requests  # type: ignore

    HF_API = "https://huggingface.co/api/models"
    resp = requests.get(f"{HF_API}/{MODEL_ID}", timeout=30)
    meta = resp.json() if resp.ok else {}
    if not resp.ok:
        print(f"[probe_model] HF API returned {resp.status_code} — metadata will be empty",
              file=sys.stderr)
except Exception as exc:  # noqa: BLE001
    print(f"[probe_model] WARNING: could not fetch HF metadata: {exc}", file=sys.stderr)
    meta = {}

print("pipeline_tag :", meta.get("pipeline_tag"))
print("library_name :", meta.get("library_name"))
print("tags         :", meta.get("tags", []))

# ---------------------------------------------------------------------------
# Step 2 — Load and inspect config
# ---------------------------------------------------------------------------

try:
    from transformers import AutoConfig  # type: ignore

    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    cfg_dict = cfg.to_dict()
except Exception as exc:  # noqa: BLE001
    print(f"[probe_model] WARNING: AutoConfig failed (trust_remote_code=False): {exc}",
          file=sys.stderr)
    cfg = None
    cfg_dict = {}

if cfg is not None:
    print("model_type          :", cfg.model_type)
    print("architectures       :", cfg.architectures)
    print("hidden_size         :", getattr(cfg, "hidden_size", "N/A"))
    print("num_hidden_layers   :", getattr(cfg, "num_hidden_layers", "N/A"))
    print("num_attention_heads :", getattr(cfg, "num_attention_heads", "N/A"))
    print("num_key_value_heads :", getattr(cfg, "num_key_value_heads", "N/A"))
    print("vocab_size          :", getattr(cfg, "vocab_size", "N/A"))
    print("max_position_embeds :", getattr(cfg, "max_position_embeddings", "N/A"))

_SPECIAL_KEYWORDS = [
    "layer_type", "ssm", "mamba", "rwkv", "moe", "expert", "conv",
    "recurrent", "delta", "vision", "image", "video", "mm_", "vlm",
    "audio", "cross_attn", "sliding", "hybrid", "flash", "rope",
    "alibi", "custom_code",
]
special_keys = [
    k for k in cfg_dict
    if any(kw in k.lower() for kw in _SPECIAL_KEYWORDS)
]
for k in special_keys:
    print(f"  {k}: {cfg_dict[k]}")

# ---------------------------------------------------------------------------
# Step 3 — Estimate parameter count
# ---------------------------------------------------------------------------

def estimate_params(cfg) -> int:
    h            = getattr(cfg, "hidden_size", 0)
    layers       = getattr(cfg, "num_hidden_layers", 0)
    vocab        = getattr(cfg, "vocab_size", 0)
    intermediate = getattr(cfg, "intermediate_size", h * 4)
    heads        = getattr(cfg, "num_attention_heads", 1)
    kv_heads     = getattr(cfg, "num_key_value_heads", heads)

    attn     = h * h + (kv_heads * (h // heads if heads else 1)) * 2 * h + h * h
    ffn      = h * intermediate * 3 if hasattr(cfg, "mlp_bias") else h * intermediate * 2
    embed    = vocab * h * 2
    per_layer = attn + ffn
    return layers * per_layer + embed


est = estimate_params(cfg) if cfg is not None else 0
print(f"[probe_model] Estimated params: ~{est / 1e9:.1f}B")

# ---------------------------------------------------------------------------
# Step 4 — Check trust_remote_code requirement
# ---------------------------------------------------------------------------

trust_remote_code_required = False
if cfg is None:
    # AutoConfig already failed without trust_remote_code — check whether it
    # succeeds with it to distinguish "needs trust_rc" from "broken model".
    try:
        from transformers import AutoConfig  # type: ignore  # noqa: F811

        AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        trust_remote_code_required = True
        # Re-load cfg with trust_remote_code for subsequent steps
        cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        cfg_dict = cfg.to_dict()
        est = estimate_params(cfg)
    except Exception:  # noqa: BLE001
        pass
else:
    try:
        from transformers import AutoConfig  # type: ignore  # noqa: F811

        AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        if "trust_remote_code" in str(exc) or "custom code" in str(exc).lower():
            trust_remote_code_required = True

print("trust_remote_code required:", trust_remote_code_required)

# ---------------------------------------------------------------------------
# Step 5 — Inspect tokenizer config
# ---------------------------------------------------------------------------

tokenizer_info = {}
try:
    from transformers import AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer_info = {
        "tokenizer_class": type(tok).__name__,
        "is_fast": tok.is_fast,
        "vocab_size": tok.vocab_size,
        "chat_template": "present" if tok.chat_template else "absent",
    }
    for k, v in tokenizer_info.items():
        print(f"{'tokenizer_' if 'tokenizer' not in k else ''}{k:<20}: {v}")
except Exception as exc:  # noqa: BLE001
    print(f"[probe_model] WARNING: tokenizer load error: {exc}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Step 6 — Detect multimodal / VLM signals
# ---------------------------------------------------------------------------

# cfg may have been re-loaded with trust_remote_code above; reload once more
# with trust_remote_code=True to get full config for VLM detection.
try:
    from transformers import AutoConfig  # type: ignore  # noqa: F811

    cfg_full = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception:  # noqa: BLE001
    cfg_full = cfg

vlm_signals = {}
if cfg_full is not None:
    vlm_signals = {
        "vision_config":  hasattr(cfg_full, "vision_config"),
        "image_token_id": hasattr(cfg_full, "image_token_id"),
        "video_token_id": hasattr(cfg_full, "video_token_id"),
        "mm_projector":   hasattr(cfg_full, "mm_projector_type"),
        "audio_config":   hasattr(cfg_full, "audio_config"),
    }

is_vlm = any(vlm_signals.values())
print("VLM signals:", vlm_signals)
print("Is VLM:", is_vlm)

# ---------------------------------------------------------------------------
# Step 7 — Check optimum-intel support
# ---------------------------------------------------------------------------

optimum_supported = False
optimum_supported_tasks = []
if cfg is not None:
    model_type = getattr(cfg, "model_type", "")
    try:
        from optimum.exporters.tasks import TasksManager  # type: ignore

        supported_tasks = TasksManager.get_supported_tasks_for_model_type(
            model_type, "openvino"
        )
        print(f"[probe_model] optimum-intel knows model_type='{model_type}': YES")
        print("Supported tasks:", supported_tasks)
        optimum_supported = True
        optimum_supported_tasks = list(supported_tasks)
    except KeyError:
        print(f"[probe_model] optimum-intel does NOT know model_type='{model_type}'")
    except ImportError:
        print("[probe_model] WARNING: optimum not installed — skipping optimum check",
              file=sys.stderr)

# ---------------------------------------------------------------------------
# Step 8 — Write model_profile.json
# ---------------------------------------------------------------------------

profile = {
    "model_id":                  MODEL_ID,
    "model_type":                getattr(cfg, "model_type", None) if cfg else None,
    "architectures":             getattr(cfg, "architectures", None) if cfg else None,
    "pipeline_tag":              meta.get("pipeline_tag"),
    "library_name":              meta.get("library_name"),
    "estimated_params_b":        round(est / 1e9, 1),
    "hidden_size":               getattr(cfg, "hidden_size", None) if cfg else None,
    "num_layers":                getattr(cfg, "num_hidden_layers", None) if cfg else None,
    "vocab_size":                getattr(cfg, "vocab_size", None) if cfg else None,
    "trust_remote_code_required": trust_remote_code_required,
    "is_vlm":                    is_vlm,
    "vlm_signals":               vlm_signals,
    "optimum_supported":         optimum_supported,
    "optimum_supported_tasks":   optimum_supported_tasks,
    "special_config_keys":       special_keys,
    "hf_tags":                   meta.get("tags", []),
    **tokenizer_info,
}

pathlib.Path("model_profile.json").write_text(
    json.dumps(profile, indent=2), encoding="utf-8"
)
print("[probe_model] Written: model_profile.json")

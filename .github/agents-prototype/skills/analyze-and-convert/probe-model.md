---
name: probe-model
description: Probe a HuggingFace model to collect its full profile — architecture class, parameter count, special requirements, tokenizer type, and known issues — before attempting conversion.
---

# Skill: Probe Model

**Trigger:** Always the first step in the `analyze-and-convert` workflow.
Produces `model_profile.json` and `probe_report.md`, consumed by all subsequent skills.

---

## Step 1 — Fetch HuggingFace Metadata

```python
import json, requests

MODEL_ID = "<model_id>"
HF_API = "https://huggingface.co/api/models"

# Model card / metadata
resp = requests.get(f"{HF_API}/{MODEL_ID}", timeout=30)
meta = resp.json() if resp.ok else {}

print("pipeline_tag :", meta.get("pipeline_tag"))
print("library_name :", meta.get("library_name"))
print("tags         :", meta.get("tags", []))
print("card_data    :", json.dumps(meta.get("cardData", {}), indent=2))
```

## Step 2 — Load and Inspect Config

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
cfg_dict = cfg.to_dict()

print("model_type          :", cfg.model_type)
print("architectures       :", cfg.architectures)
print("hidden_size         :", getattr(cfg, "hidden_size", "N/A"))
print("num_hidden_layers   :", getattr(cfg, "num_hidden_layers", "N/A"))
print("num_attention_heads :", getattr(cfg, "num_attention_heads", "N/A"))
print("num_key_value_heads :", getattr(cfg, "num_key_value_heads", "N/A"))
print("vocab_size          :", getattr(cfg, "vocab_size", "N/A"))
print("max_position_embeds :", getattr(cfg, "max_position_embeddings", "N/A"))

# Detect special architecture markers
special_keys = [k for k in cfg_dict if any(kw in k.lower() for kw in [
    "layer_type", "ssm", "mamba", "rwkv", "moe", "expert", "conv",
    "recurrent", "delta", "vision", "image", "video", "mm_", "vlm",
    "audio", "cross_attn", "sliding", "hybrid", "flash", "rope",
    "alibi", "custom_code",
])]
for k in special_keys:
    print(f"  {k}: {cfg_dict[k]}")
```

## Step 3 — Estimate Parameter Count

```python
from transformers import AutoConfig

# Rough estimation from config (no weights download needed)
def estimate_params(cfg):
    h = getattr(cfg, "hidden_size", 0)
    layers = getattr(cfg, "num_hidden_layers", 0)
    vocab = getattr(cfg, "vocab_size", 0)
    intermediate = getattr(cfg, "intermediate_size", h * 4)
    heads = getattr(cfg, "num_attention_heads", 1)
    kv_heads = getattr(cfg, "num_key_value_heads", heads)

    # attn (Q + K + V + O)
    attn = h * h + (kv_heads * (h // heads)) * 2 * h + h * h
    # ffn (gate + up + down for LLaMA-style, or 2-linear for BERT-style)
    ffn = h * intermediate * 3 if hasattr(cfg, "mlp_bias") else h * intermediate * 2
    # embeddings
    embed = vocab * h * 2  # input + output (tied)
    per_layer = attn + ffn
    total = layers * per_layer + embed
    return total

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
est = estimate_params(cfg)
print(f"Estimated params: ~{est/1e9:.1f}B")
```

## Step 4 — Check trust_remote_code Requirement

```python
# Try loading without trust_remote_code first
try:
    from transformers import AutoConfig
    AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=False)
    trust_remote_code_required = False
except Exception as e:
    if "trust_remote_code" in str(e) or "custom code" in str(e).lower():
        trust_remote_code_required = True
    else:
        trust_remote_code_required = False  # different error
print("trust_remote_code required:", trust_remote_code_required)
```

## Step 5 — Inspect Tokenizer Config

```python
from transformers import AutoTokenizer
import json

try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("tokenizer_class  :", type(tok).__name__)
    print("is_fast          :", tok.is_fast)
    print("vocab_size       :", tok.vocab_size)
    print("chat_template    :", "present" if tok.chat_template else "absent")
except Exception as e:
    print(f"Tokenizer load error: {e}")
```

## Step 6 — Detect Multimodal / VLM Signals

```python
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)

vlm_signals = {
    "vision_config": hasattr(cfg, "vision_config"),
    "image_token_id": hasattr(cfg, "image_token_id"),
    "video_token_id": hasattr(cfg, "video_token_id"),
    "mm_projector": hasattr(cfg, "mm_projector_type"),
    "audio_config": hasattr(cfg, "audio_config"),
}
is_vlm = any(vlm_signals.values())
print("VLM signals:", vlm_signals)
print("Is VLM:", is_vlm)
```

## Step 7 — Check optimum-intel Support

```python
from optimum.exporters.tasks import TasksManager

model_type = cfg.model_type
try:
    supported_tasks = TasksManager.get_supported_tasks_for_model_type(
        model_type, "openvino"
    )
    print(f"optimum-intel knows model_type='{model_type}': YES")
    print("Supported tasks:", supported_tasks)
    optimum_supported = True
except KeyError:
    print(f"optimum-intel does NOT know model_type='{model_type}'")
    optimum_supported = False
```

## Step 8 — Write model_profile.json

```python
import json

profile = {
    "model_id": MODEL_ID,
    "model_type": cfg.model_type,
    "architectures": cfg.architectures,
    "pipeline_tag": meta.get("pipeline_tag"),
    "library_name": meta.get("library_name"),
    "estimated_params_b": round(est / 1e9, 1),
    "hidden_size": getattr(cfg, "hidden_size", None),
    "num_layers": getattr(cfg, "num_hidden_layers", None),
    "vocab_size": getattr(cfg, "vocab_size", None),
    "trust_remote_code_required": trust_remote_code_required,
    "is_vlm": is_vlm,
    "vlm_signals": vlm_signals,
    "optimum_supported": optimum_supported,
    "special_config_keys": special_keys,
    "hf_tags": meta.get("tags", []),
}

with open("model_profile.json", "w") as f:
    json.dump(profile, f, indent=2)

print("Written: model_profile.json")
```

## Output

| File | Contents |
|------|----------|
| `model_profile.json` | Structured model profile consumed by subsequent skills |

Key fields used downstream:

| Field | Consumed by |
|-------|-------------|
| `optimum_supported` | `try-conversion` — decides which export path to use |
| `trust_remote_code_required` | `try-conversion` — adds flag if needed |
| `is_vlm` | `classify-failure` — routes to GenAI/VLM specialists |
| `estimated_params_b` | `build-report` — included in diagnostics for orchestrator |
| `special_config_keys` | `classify-failure` — identifies SSM/MoE/recurrent patterns |

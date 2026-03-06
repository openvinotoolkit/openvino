# Skill: Create New Model Configuration

**Trigger:** User asks to add export support for a model type missing from `model_configs.py`.

## Prerequisites

- Run **optimum_bootstrap** skill first.

## Steps

1. **Read the SKILL.md** from the bootstrapped optimum-intel clone for the full config class pattern.

2. **Analyse the model architecture:**
   ```python
   from transformers import AutoModelForCausalLM
   import torch
   model = AutoModelForCausalLM.from_pretrained("<MODEL_ID>", torch_dtype=torch.bfloat16)
   for name, module in model.named_modules():
       print(f"{name}: {type(module).__name__}")
   ```

3. **Identify the closest existing config class** in `optimum/exporters/openvino/model_configs.py` - look for a model with similar architecture (e.g., LLaMA-like, GPT-like, encoder-decoder).

4. **Create the config class** following the naming convention `<ModelType>OpenVINOConfig`.

5. **Register the model type** in the task manager mappings.

6. **Test the export** with the model or a tiny version of it.

## Conventions

- **Config class naming:** `<ModelType>OpenVINOConfig` (e.g., `LlamaOpenVINOConfig`, `QwenOpenVINOConfig`).

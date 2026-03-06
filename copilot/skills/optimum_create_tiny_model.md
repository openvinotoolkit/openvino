# Skill: Create Tiny Model for Testing

**Trigger:** User asks to create a small/tiny version of a model for CI testing.

## Prerequisites

- Run **optimum_bootstrap** skill first.

## Steps

1. Load the original model config:
   ```python
   from transformers import AutoConfig
   config = AutoConfig.from_pretrained("<MODEL_ID>")
   print(config)  # Inspect architecture fields
   ```

2. Create a minimal config by overriding size parameters:
   ```python
   tiny_config = AutoConfig.from_pretrained("<MODEL_ID>")
   tiny_config.num_hidden_layers = 1
   tiny_config.hidden_size = 64
   tiny_config.intermediate_size = 256
   tiny_config.num_attention_heads = 2
   tiny_config.num_key_value_heads = 2
   ```
   - For models with `layer_types` (hybrid architectures), keep a representative subset.
   - Adjust vocab size if it dominates parameter count: `tiny_config.vocab_size = min(tiny_config.vocab_size, 1000)`.
   - Handle architecture-specific fields (e.g., `num_experts`, `num_experts_per_tok` for MoE models).

3. Instantiate and save:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   model = AutoModelForCausalLM.from_config(tiny_config)
   model.save_pretrained("tiny_model")
   tokenizer = AutoTokenizer.from_pretrained("<MODEL_ID>")
   tokenizer.save_pretrained("tiny_model")
   ```

4. Verify size constraint:
   ```python
   import os
   total = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fns in os.walk("tiny_model") for f in fns)
   assert total < 1_000_000_000, f"Tiny model too large: {total / 1e9:.2f} GB"
   ```

5. Test that it exports cleanly:
   ```bash
   optimum-cli export openvino --model tiny_model --task text-generation-with-past --weight-format fp16 ov_tiny_model
   ```

## Conventions

- Target size < 1 GB.
- Typical dims: `num_hidden_layers=1`, `hidden_size=64`, `num_attention_heads=2`.

# Skill: Add Full New Model Support

**Trigger:** User asks to add complete support for a new model architecture in optimum-intel.

## Prerequisites

- Run **optimum_bootstrap** skill first.
- Study the reference PR (#1569 Afmoe) for a canonical example:
  ```bash
  cd /tmp/optimum-intel
  git fetch origin pull/1569/head:pr-1569
  git diff main...pr-1569 --name-status
  git diff main...pr-1569 -- optimum/exporters/openvino/
  ```

## Steps

Execute in order:

### 1. Model Architecture Analysis

Identify model family, block types, special patterns (MoE, linear attention,
cross-attention, etc.).

### 2. Update `optimum/exporters/openvino/model_configs.py`

Add a new config class inheriting from the closest existing one.
→ Use **optimum_create_model_config** skill for details.

### 3. Update `optimum/exporters/openvino/model_patcher.py` (if needed)

Add patching functions for `torch.jit.trace`-incompatible patterns. Principles:
- Replace Python control flow that depends on runtime tensor values with vectorised torch ops.
- Replace `for` loops over experts (MoE) with batched matrix multiplications.
- Ensure all code paths produce the same graph regardless of input data.

### 4. Create / Update Tests

| Test file | What to add |
|-----------|-------------|
| `tests/openvino/test_decoder.py` | Export and inference validation |
| `tests/openvino/test_export.py` | Export configurations |
| `tests/openvino/test_exporters_cli.py` | CLI tests |
| `tests/openvino/test_quantization.py` | Add to `SUPPORTED_ARCHITECTURES_WITH_AUTO_COMPRESSION`, update `_ARCHITECTURES_TO_EXPECTED_INT8` |
| `tests/openvino/utils_tests.py` | Define test model IDs |

### 5. Update Documentation

Add the new `model_type` (first letter capitalised) to `docs/source/openvino/models.mdx`.

### 6. Verify

```bash
cd /tmp/optimum-intel
pip install -e ".[tests]"
pytest tests/openvino/test_export.py -k "<model_type>" -v
pytest tests/openvino/test_decoder.py -k "<model_type>" -v
```

## Key Files in optimum-intel

| File | Purpose |
|------|---------|
| `optimum/exporters/openvino/model_configs.py` | Model config classes for OV export |
| `optimum/exporters/openvino/model_patcher.py` | Model patching for trace-safe conversion |
| `optimum/exporters/tasks.py` | Task manager - model type ↔ task registration |
| `tests/openvino/test_decoder.py` | Decoder model export + inference tests |
| `tests/openvino/test_export.py` | Export configuration tests |
| `tests/openvino/test_exporters_cli.py` | CLI export tests |
| `tests/openvino/test_quantization.py` | Quantisation workflow tests |
| `tests/openvino/utils_tests.py` | Test model IDs and helper constants |
| `docs/source/openvino/models.mdx` | Supported models documentation |

## Conventions

- **Patching safety:** All patches must produce identical torch graphs regardless of input data - no Python `if/for` that depends on tensor values.
- **Vectorise MoE:** Replace per-expert loops with batched `torch.bmm` over stacked weight matrices.
- **Tiny models for CI:** Always < 1 GB (`num_hidden_layers=1`, `hidden_size=64`, `num_attention_heads=2`).

## External References

- **torch.jit.trace docs:** https://pytorch.org/docs/stable/generated/torch.jit.trace.html
- **Optimum Intel docs:** https://huggingface.co/docs/optimum-intel/en/index
- **OpenVINO documentation:** https://docs.openvino.ai/2025/index.html

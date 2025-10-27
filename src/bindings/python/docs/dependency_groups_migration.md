# OpenVINO UV Compatibility Guide

## Overview

OpenVINO is now compatible with UV package manager and modern PEP 735 dependency groups. 

**Important Migration Notice:**
- **Recommended**: Use dependency groups defined in `pyproject.toml` with pip 25.1+ or UV
- **Deprecated**: `requirements.txt` and `constraints.txt` files will be removed after a migration period
  


Dependencies are available through both traditional requirements.txt files (deprecated and set to be removed in OpenVINO XX.X) and modern PEP 735 dependency groups (recommended).



## Why migrate to UV?

UV is recommended for dependency management because:

- **Speed**: UV installs dependencies much faster than pip by using a Rust-based resolver and installer.
- **Reliability**: UV provides more consistent and reproducible installs, reducing dependency conflicts and resolver errors.
- **PEP 735 Support**: UV natively supports dependency groups, making it easier to manage complex project requirements.
- **Modern Python Packaging**: UV is designed for modern workflows and works seamlessly with `pyproject.toml`.

While pip 25.1+ also supports dependency groups, UV is preferred for its performance and reliability, especially in CI/CD and developer environments.

## Mapping: Requirements Files → Dependency Groups

This table helps you migrate from requirements files to dependency groups:

| Requirements File | Dependency Group | Purpose |
|-------------------|------------------|---------|
| `src/bindings/python/requirements.txt` | `runtime` | Runtime dependencies |
| `src/bindings/python/wheel/requirements-dev.txt` | `dev_wheel` | Development/build tools |
| `src/bindings/python/requirements_test.txt` | `tests_pyapi` | Python API tests & linting |
| `src/frontends/onnx/tests/requirements.txt` | `tests_onnx_frontend` | ONNX frontend tests |
| `src/frontends/paddle/tests/requirements.txt` | `tests_paddle_frontend` | PaddlePaddle frontend tests |
| `src/frontends/tensorflow/tests/requirements.txt` | `tests_tensorflow_frontend` | TensorFlow frontend tests |
| `src/frontends/tensorflow_lite/tests/requirements.txt` | `tests_tensorflow_lite_frontend` | TF Lite frontend tests |
| `tests/layer_tests/requirements.txt` | `tests_layer` | Layer tests |
| `tests/e2e_tests/requirements.txt` | `tests_e2e` | End-to-end tests |
| `tests/conditional_compilation/requirements.txt` | `tests_conditional_compilation` | Conditional compilation tests |
| `tests/model_hub_tests/performance_tests/requirements.txt` | `tests_model_hub_performance` | Model hub performance tests |
| `tests/model_hub_tests/tensorflow/requirements.txt` | `tests_model_hub_tensorflow` | TensorFlow model hub tests |
| `tests/samples_tests/smoke_tests/requirements.txt` | `tests_samples` | Sample tests |
| `tests/stress_tests/scripts/requirements.txt` | `tests_stress` | Stress tests |
| `tests/time_tests/scripts/requirements.txt` | `tests_time` | Time tests |
| `tests/time_tests/test_runner/requirements.txt` | `tests_time_runner` | Time test runner |
| `tests/llm/requirements.txt` | `tests_llm` | LLM tests |
| `src/tests/test_utils/functional_test_utils/layer_tests_summary/requirements.txt` | `tests_utils_layer_summary` | Layer summary utilities |
| `src/bindings/python/src/openvino/preprocess/torchvision/requirements.txt` | `torchvision_preprocessing` | Torchvision preprocessing |
| `src/plugins/intel_cpu/tools/dump_check/requirements.txt` + `src/plugins/intel_cpu/tools/aggregate-average-counters/requirements.txt` | `cpu_tools` | CPU profiling tools (merged) |
| `docs/requirements.txt` | `docs` | Documentation |
| `samples/python/benchmark/bert_benchmark/requirements.txt` | `samples_bert_benchmark` | BERT benchmark sample |
| `samples/python/hello_classification/requirements.txt` | `samples_hello_classification` | Hello classification sample |
| `samples/python/classification_sample_async/requirements.txt` | `samples_classification` | Classification sample |
| `samples/python/hello_reshape_ssd/requirements.txt` | `samples_hello_reshape_ssd` | Hello reshape SSD sample |

### Migration Examples

**Before (requirements files):**
```bash
pip install -r src/bindings/python/wheel/requirements-dev.txt
pip install -r tests/layer_tests/requirements.txt
pip install -r src/frontends/onnx/tests/requirements.txt
```

**After (dependency groups):**
```bash
# Using UV (recommended)
uv pip install --group dev_wheel --group tests_layer --group tests_onnx_frontend

# Using pip 25.1+
pip install --group dev_wheel --group tests_layer --group tests_onnx_frontend
```


| Old Command | New Command | Description |
|-------------|-------------|-------------|
| `pip install -r src/bindings/python/wheel/requirements-dev.txt` | `uv pip install --group dev_wheel` | Install build tools |
| `pip install -r src/bindings/python/requirements_test.txt` | `uv pip install --group tests_pyapi` | Install test/lint tools |
| `pip install -r src/bindings/python/requirements.txt` | `uv pip install --group runtime` | Install runtime deps |

## Sources

- Check the [official UV documentation](https://docs.astral.sh/uv/)
- See [PEP 735](https://peps.python.org/pep-0735/) for dependency groups specification

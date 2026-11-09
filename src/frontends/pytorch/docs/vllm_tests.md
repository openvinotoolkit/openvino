# Testing the torch.compile and vLLM paths

| Suite | Location | Covers |
|---|---|---|
| vLLM tests | [tests/vllm_tests](../../../../tests/vllm_tests/) | vLLM + OpenVINO end to end, and unit tests of the plugin package |
| PyTorch frontend tests | [test_torch_frontend.py](../../../../tests/layer_tests/py_frontend_tests/test_torch_frontend.py) | Decoder `default_float_type`, `CanonicalizeFloatPrecision`, `MarkCompressedWeightsCast` |
| Transformation tests | [normalize_vllm_rope_test.cpp](../../../common/transformations/tests/common_optimizations/normalize_vllm_rope_test.cpp), [vllm_gating_test.cpp](../../../common/transformations/tests/common_optimizations/vllm_gating_test.cpp) | `NormalizeVLLMRoPE` and its `vllm_model` gate (`ov_transformations_tests`) |
| CPU plugin tests | `ov_cpu_unit_tests`, `ov_cpu_func_tests` | MLP and QKV projection fusions on the flattened `[tokens, hidden]` graphs this path produces, and a bf16 attention-mask subgraph |

## vLLM tests

The suite needs `vllm-cpu` and its own validated PyTorch, which can't be installed together with the other test
requirements, so it has its own requirements file,
[tests/requirements_vllm](../../../../tests/requirements_vllm). Use a separate virtual environment:

```sh
python3 -m venv venv_vllm && . venv_vllm/bin/activate
pip install -r tests/requirements_vllm
pip install <openvino wheel built from this tree>
OV_LAZY_KVCACHE=1 python -m pytest tests/vllm_tests -m precommit -v
```

* End-to-end tests download `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from the Hugging Face Hub. Set
  `HF_HUB_OFFLINE=1` once it is cached.
* `OV_LAZY_KVCACHE=1` allocates the KV-cache pool lazily; without it the suite needs considerably more memory.
* Pin the run to one NUMA node (`numactl --cpunodebind=N --membind=N` and
  `VLLM_CPU_OMP_THREADS_BIND=<cores of N>`) on multi-socket machines.
* `conftest.py` calls `preset.set_pre_import_env()` before vLLM is imported and provides `new_openvino_llm()`,
  which builds a bf16 TinyLlama with the compilation config the backend requires. Each test builds and deletes
  its own instance.

| File | Kind | Covers |
|---|---|---|
| `test_run.py` | End to end | OpenVINO backend output vs vLLM eager |
| `test_advanced_features.py` | End to end | logprobs, prompt-length changes and continuous batching, grammar, beam search, speculative decoding, prefix caching, logits processors |
| `test_tiny_models.py` | End to end | Tiny random-weight MoE and Gated DeltaNet models vs eager |
| `test_reload_weights.py`, `test_lm_head_reload.py` | End to end | The OV model and lm_head pick up weights changed after load |
| `test_attention_support.py` | Unit | Refusing attention features the PagedAttention rewrite can't express |
| `test_pa_partitioning.py` | Unit | Attention calls always land in an OV partition; KV-sharing pairs stay together |
| `test_pa_input_binding.py`, `test_fast_infer.py` | Unit | Side-channel and fast-infer input binding |
| `test_fused_sampler.py` | Unit | The OV-fused sampler |
| `test_omp_wait_policy.py`, `test_bench_run.py` | Unit | Worker OpenMP environment; benchmark script arguments and timing |

New tests that should run in CI need the `precommit` marker. Prefer unit tests on hand-built FX graphs or
stand-in objects where no model is needed. For end-to-end checks against eager, tiny random models need the
tie-aware token comparison of `test_tiny_models.py`, because random weights give flat logits.

## CI

[job_vllm_tests.yml](../../../../.github/workflows/job_vllm_tests.yml) runs `pytest -m precommit` on the installed
test package in the Ubuntu 22.04 and 24.04 workflows, when the PyTorch frontend is affected and the run isn't a
scheduled nightly. It installs `libnuma1` and the Python headers (vLLM's own sampler compiles a C++ extension
through Inductor), then `requirements_vllm`, and sets `OV_LAZY_KVCACHE=1`.

## PyTorch frontend tests

```sh
cd tests/layer_tests
python -m pytest py_frontend_tests/test_torch_frontend.py -k "canonicalize_float_precision or default_float_type or weights_cast or weights_trailing_cast"
```

They need the usual layer-test requirements and no vLLM. See
[narrow default float type](default_float_type.md#tests) for how the rounding cases are organized.

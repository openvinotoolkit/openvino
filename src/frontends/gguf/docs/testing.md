# Building and validating the GGUF frontend

Use the repository [build prerequisites](../../../../docs/dev/build.md). GGUF defaults to enabled
(`ENABLE_OV_GGUF_FRONTEND=ON`); C++ tests require `ENABLE_TESTS=ON` and a shared-library build.
This Linux single-configuration example puts binaries and libraries in an explicit directory:

```sh
cmake -S . -B build-gguf -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OV_GGUF_FRONTEND=ON -DENABLE_TESTS=ON -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_RUNTIME_OUTPUT_DIRECTORY="$PWD/build-gguf/bin" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$PWD/build-gguf/bin"
cmake --build build-gguf --target ov_gguf_frontend_tests ov_gguf_architecture_library_tests --parallel
./build-gguf/bin/ov_gguf_frontend_tests
./build-gguf/bin/ov_gguf_architecture_library_tests
```

For an existing build, use its configured output directory. OpenVINO normally writes under
the source tree's `bin/<architecture>/<configuration>`, not necessarily under the CMake build tree.
Multi-configuration generators can add a configuration subdirectory.

## Select coverage

| Change | Coverage |
|---|---|
| Converter or `op_case` | Relevant `GGUFOps` tests, then the full frontend binary for the operation-coverage gate |
| Architecture/shared decoder blocks | `GGUFArchConversion`, `GGUFArchitectureAccuracy`, relevant adaptation/state tests |
| mmproj | `GGUFMMProj`, `GGUFMMProjAccuracy`, `GGUFMMProjDynamicAccuracy`; relevant real-checkpoint tests |
| Extension API or packaging | Builder/extension tests plus the separate architecture-library binary and actual shipped plugin |
| Weight conversion | Weight/dequantization tests and checkpoint comparisons with recorded precision settings |

Use `--gtest_list_tests` for exact parameterized names; filter while iterating, for example
`--gtest_filter='GGUFOps.Scale*'`. A narrowed filter skips the operation-coverage gate. That gate
runs in environment teardown, so inspect the process exit status even if no individual test failed.
Shared changes require the affected architecture/projector numerical and structural regressions.

## Fixtures and oracles

Small NPZ/NPY numerical references are committed and work offline. Missing numerical references
fail their tests. Architecture header fixtures are generated separately; if none are present,
`GGUFArchConversion` skips, while a partially generated set fails. Generate them before claiming
architecture fingerprint coverage.

The generator requires NumPy and the `gguf` Python package (llama.cpp's `gguf-py`) in the active
environment, plus Git, CMake, and a C/C++ compiler. `--fetch` does not install Python dependencies.

```sh
python3 src/frontends/gguf/tests/gen_arch_fixtures.py --fetch
```

This fetches/builds the pinned llama.cpp generator and writes headers plus the manifest under
`tests/test_data/arch_fixtures`. Review manifest changes. `--llama-src /path/to/llama.cpp` uses an
existing checkout; see the script's pinned revision and prerequisites. The optional Python
fingerprint suite also skips unless `GGUF_FINGERPRINT_MODELS=name=path[=hash];...` is set.
Fingerprints are structural checks, not proof of numerical equivalence.

Regenerate numerical fixtures using their own contracts and reference revisions:

- [Decoder/embedding references](../tests/test_data/arch_accuracy/README.md), including the
  separately recorded Mamba 2/Nemotron-H oracle revision.
- [mmproj references](../tests/test_data/mmproj_accuracy/README.md), including raw/adapted output
  checks and multiple sizes on one compiled model.
- [Single-operation oracles](debugging_accuracy.md#the-ggml-cpu-oracle) for layout-sensitive operations.

## Real checkpoints

Use a Python environment containing the matching OpenVINO build with the GGUF frontend.
From the repository root, install the test dependencies and run the selected model-hub scope:

```sh
python3 -m pip install -r tests/requirements_gguf
TEST_DEVICE=CPU PYTHONPATH="$PWD/tests/model_hub_tests${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest tests/model_hub_tests/gguf -m precommit -v
```

Use `-m nightly` for the larger list, or select `test_gguf.py` / `test_gguf_mmproj.py`.
Decoder model-hub tests check finite, shaped outputs and state updates; they do not compare
tokens/logits against a reference. The native reference-token replay instead uses
`OV_GGUF_ACCURACY_DATA` (and `OV_GGUF_EMBEDDING_DATA` for `llama-embed`) as described in the
[decoder fixture README](../tests/test_data/arch_accuracy/README.md). Run this separately: the
variable replaces the selected synthetic fixture with a local checkpoint.

mmproj model-hub tests compare real checkpoint encoders against llama.cpp on an F32 copy of the
represented weights. They use generated test inputs, not a full media application. Downloads,
a C/C++ compiler, Git, CMake, and Ninja are needed. Reference builds are cached under
`GGUF_LLAMA_CPP_CACHE` (default `~/.cache/openvino_gguf_llama_cpp`). `GGUF_MMPROJ_ORACLE` selects a
prebuilt oracle, but the pinned checkout is still needed for `gguf-py`. `HF_HUB_CACHE` controls
checkpoint caching. To use local files, point `GGUF_MMPROJ_LIST` at a replacement precommit list:

```text
gemma3,,/absolute/path/mmproj-gemma3.gguf
voxtral,,/absolute/path/mmproj-voxtral.gguf
```

The list format, skip reasons, and tolerance overrides live in
[precommit](../../../../tests/model_hub_tests/gguf/gguf_mmproj_precommit) and
[nightly](../../../../tests/model_hub_tests/gguf/gguf_mmproj_nightly).

## Acceptance

Keep registration, structural conversion, numerical agreement, and application functionality
as separate evidence. A new architecture needs a nonzero oracle fixture and a real-checkpoint
comparison; exercise relevant positions, sizes, cache appends/resets, and adaptation paths.

| Check | Acceptance and limit |
|---|---|
| Native decoder F32 fixture | Full-logit normalized MSE below `1e-5`; F32 inference, F16 KV, activation quantization disabled |
| Native real decoder replay | Same first prediction and at least 90% matching greedy choices over prefill plus 12 reference-token steps; record full-logit errors |
| mmproj fixture | Raw and adapted embeddings, normalized MSE below `1e-5`; dynamic cases reuse one compiled model |
| mmproj real checkpoint | CPU NMSE below `1e-5` unless an explicit model-list override documents the reason; other devices are not calibrated to the CPU limit |

Use [quantization controls](quantization.md#accuracy-controls) consistently. Report checkpoint,
revisions, device, precision, inputs, thresholds, skips, and failures. A represented-F32 reference
does not erase a failed comparison against quantized CPU arithmetic, and coherent generated text
does not establish encoder accuracy or broad answer quality.

CI runs the two C++ binaries through [Smart CI's GGUF component gate](../../../../.github/workflows/job_cxx_unit_tests.yml).
The jobs invoke GTest directly and upload XML reports. Fixture availability and
conversion expectations are checked by the architecture suite itself. Run the
same binaries locally with `--gtest_filter` while editing; use GHA for broader
coverage and inspect test counts and skips in its artifacts.
[Model-hub CI](../../../../.github/workflows/job_gguf_models_tests.yml) runs checkpoint lists;
[llama.cpp compatibility CI](../../../../.github/workflows/job_gguf_llamacpp_validation.yml) records
the backend revision, operator tests, state scenarios, and exclusions. These validate different contracts.

[GenAI acceptance CI](../../../../.github/workflows/job_gguf_genai_validation.yml)
uses this workflow's OpenVINO package and wheels and a pinned companion GenAI
commit. Its pytest harness builds an independent pinned llama.cpp CPU oracle,
converts pinned tiny checkpoints, and runs square/JPEG, SDPA/PA, single/multiple
media, video/audio, chat, cancellation/reset and beam-search cases. Separate
checks cover Optimum directory/map loading and a real Qwen3.5 pure Q4_0 fixture.
Acceptance reports record required coverage and source/precision provenance;
Q4_K_M is excluded from accuracy gates pending the plugin fix. Known Gemma4
window and Unified prefix-cache parity gaps remain documented in [mmproj.md](mmproj.md).

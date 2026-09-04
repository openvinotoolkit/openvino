---
name: ov-update-pytorch-version
description: >
  Upgrade the PyTorch version used by OpenVINO tests (torch / torchvision / torchaudio) and
  resolve fallout — missing operator translators, new functionalized `*_copy` aten ops,
  decomposition changes, FX-only tests failing in TorchScript mode, and accuracy regressions
  caused by stricter typing. Use when the user asks to "bump torch", "update pytorch to X.Y",
  "upgrade torch tests", or when pytorch_tests / model_hub pytorch tests fail after a torch
  version change. Do not use for: enabling a single new PyTorch operator unrelated to a version
  bump, GenAI / Optimum upgrades, or plugin-level numerical bugs unrelated to the frontend.
---

# Update PyTorch Version in OpenVINO

End-to-end procedure for bumping the PyTorch version used by OpenVINO layer tests and the PyTorch frontend, and fixing the regressions that typically follow.

## Step 0: Confirm Scope

Before editing anything, confirm with the user:

- Target versions for `torch`, `torchvision`:
  - `torchvision` minor = torch minor **+ 15** — e.g. torch `2.9.0` ↔ torchvision `0.24.0`, torch `2.12.0` ↔ torchvision `0.27.0`. Always confirm against the official release matrix.
  - `torchaudio` is **not** pinned by [tests/requirements_pytorch](../../../tests/requirements_pytorch): it is unused by these tests and PyTorch removed it from the official installation instructions starting with the 2.8 release. **Do not re-add it.** If a future task genuinely needs torchaudio, note that it is in a maintenance phase and its latest release can lag torch by one or more minors (e.g. torch `2.12.0` ↔ torchaudio `2.11.0`), so `torchaudio==<torch version>` may not exist — look up the latest available wheel at `https://download.pytorch.org/whl/cpu/torchaudio/`. Recent torchaudio wheels carry **no** `Requires-Dist: torch` pin, so a slightly older torchaudio installs cleanly alongside a newer torch.
- Whether [tests/model_hub_tests/pytorch/envs/compile_gptq.txt](../../../tests/model_hub_tests/pytorch/envs/compile_gptq.txt) (pinned at `torch==2.3.1`, `torchaudio==2.3.1` for auto-gptq) should be touched. **Default: leave it alone.**
- Both the default TorchScript path **and** `PYTORCH_TRACING_MODE=EXPORT` (FX) path must be validated. Skip the FX run only if the user explicitly opts out.

## Step 1: Update Version Pins

Edit the following files. Keep `~=` style in `constraints.txt`, exact `==` in `requirements_*`. `A.B` is the torchvision version (torch minor + 15):

- [tests/constraints.txt](../../../tests/constraints.txt) — `torch~=X.Y.0`, `torchvision~=A.B.0` (constraints.txt does not pin torchaudio)
- [tests/requirements_pytorch](../../../tests/requirements_pytorch) — `torch==X.Y.0`, `torchvision==A.B.0` (no torchaudio — see Step 0)
- `tests/model_hub_tests/pytorch/envs/*.txt` — bump in lockstep **except** `compile_gptq.txt` unless the user confirms.

Install into the active environment:

```bash
pip3 install --index-url https://download.pytorch.org/whl/cpu --upgrade \
    torch==X.Y.0 torchvision==A.B.0
```

## Step 2: Run the Layer Tests in Parallel

Always test against a freshly built `openvino_pytorch_frontend`. A site-wide install (e.g. under `~/.local`) can shadow the dev build — explicitly point at the build output. Set `OV_REPO` to the OpenVINO checkout root and `OV_BUILD_BIN` to its build artifact dir (typically `$OV_REPO/bin/intel64/Release`):

```bash
cd "$OV_REPO/tests/layer_tests"
export PYTHONPATH="$OV_BUILD_BIN/python"
export LD_LIBRARY_PATH="$OV_BUILD_BIN"
export TEST_DEVICE=CPU TEST_PRECISION=FP32
```

### No existing build? Configure a minimal one first

If `$OV_REPO` has no build directory yet, don't build the whole product — configure a Release build with only what the layer tests need, so the first build stays reasonably fast:

```bash
cd "$OV_REPO" && git submodule update --init --recursive --jobs 8
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DENABLE_PYTHON=ON -DENABLE_PYTHON_PACKAGING=ON \
  -DENABLE_TESTS=OFF -DENABLE_SAMPLES=OFF -DENABLE_TEMPLATE=OFF \
  -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF \
  -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF \
  -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF \
  -DENABLE_JS=OFF -DENABLE_WHEEL=OFF
cmake --build build --parallel --target openvino_pytorch_frontend py_pytorch_frontend pyopenvino openvino_intel_cpu_plugin
```

Using `cmake -S . -B build` (rather than `mkdir build && cd build && cmake ..`) keeps the working directory fixed at the repo root, and driving the build through `cmake --build --target ...` (rather than invoking `ninja` directly) keeps this platform-agnostic across generators.

Enable `ccache` (`-DCMAKE_C_COMPILER_LAUNCHER=ccache`/`CXX` variant) whenever it's installed — a warm cache from a prior build of the same or a nearby commit cuts a from-scratch build from tens of minutes to a few minutes.

Four targets are needed, and it's easy to build the wrong subset:

- `pyopenvino` — the core Python bindings (`openvino._pyopenvino`). Note the target is `pyopenvino`, **not** `_pyopenvino` (that name doesn't exist and the build will error on it).
- `py_pytorch_frontend` — the pybind module the layer tests actually import (`openvino.frontend.pytorch.py_pytorch_frontend`, via `TorchScriptPythonDecoder`/`fx_decoder`). This is **separate** from `openvino_pytorch_frontend` (the C++ frontend library) — building only the C++ target still leaves the tests raising `ImportError: OpenVINO PyTorch frontend is not available ... No module named 'openvino.frontend.pytorch.py_pytorch_frontend'`.
- `openvino_pytorch_frontend` — the C++ frontend itself; rebuild after any translator/op_table change (see Step 3).
- `openvino_intel_cpu_plugin` — required whenever `TEST_DEVICE=CPU` (the layer test default). Without it, every single test fails at inference time with `Device with "CPU" name is not registered in the OpenVINO Runtime` — don't mistake this for a translator regression.

If **every** test fails immediately (e.g. `pytest` output is a wall of `F`s with no per-test variation, or an `ImportError`/`Device ... not registered` in the first failure), the build/environment is incomplete — stop and fix that before triaging with Step 3's buckets, which assume the harness itself works and only individual translators/tests are broken.

The pytorch layer tests use two pytest markers to gate which path a test runs on:

- `-m precommit` — TorchScript path (default tracing mode). Use this to validate the TS frontend.
- `-m precommit_torch_export` — `torch.export` / FX path. Requires `PYTORCH_TRACING_MODE=EXPORT`; the test runner reads that env var to switch the test class behaviour.

Both runs are required after a version bump:

```bash
# TorchScript path
python3 -m pytest pytorch_tests/ -m precommit -n auto --tb=line -q \
    2>&1 | tee /tmp/pt_run_ts.log | tail -5

# torch.export (FX) path
PYTORCH_TRACING_MODE=EXPORT \
python3 -m pytest pytorch_tests/ -m precommit_torch_export -n auto --tb=line -q \
    2>&1 | tee /tmp/pt_run_fx.log | tail -5
```

Collect unique failing test files from either log:

```bash
grep -E "^FAILED" /tmp/pt_run_ts.log /tmp/pt_run_fx.log | sed 's/\[.*//' | sort -u
```

Run one failing test verbosely to see the real traceback before grouping fixes (add `PYTORCH_TRACING_MODE=EXPORT` for FX-side reproduction):

```bash
python3 -m pytest pytorch_tests/test_<name>.py -k <case> -x --tb=long
```

## Step 3: Triage Failures

Sort failures into these buckets. The first three are the common ones for any minor torch bump.

> **Rebuild after any C++ change.** Buckets A and B both modify the PyTorch frontend; re-run before re-testing:
>
> ```bash
> cmake --build "$OV_REPO/build" --target openvino_pytorch_frontend -j$(nproc)
> ```

### Bucket A — Newly-emitted aten ops with no translator

Each torch release tends to lower more ops to new aten variants. The `*_copy` family from functionalization (e.g. `aten::select_copy`, `aten::view_copy`, `aten::squeeze_copy`, `aten::expand_copy`, `aten::permute_copy`, `aten::as_strided_copy`, `aten::split_with_sizes_copy`, `aten::unsqueeze_copy`) is one common pattern, but the bucket also covers any other freshly-introduced op (renames, new overloads, ops promoted out of decomposition, etc.). Symptom: log line like `No translator found for aten::<name>`.

Decision tree:

1. **Semantically equivalent to an existing op** (typical for `_copy` variants — OV graphs are functional, so the copy is a no-op) → alias to the existing translator in the **TorchScript** table (`aten::*` keys) of [src/frontends/pytorch/src/op_table.cpp](../../../src/frontends/pytorch/src/op_table.cpp):

   ```cpp
   {"aten::select_copy",            op::quantizable_op<op::translate_select>},
   {"aten::view_copy",              op::quantizable_op<op::translate_reshape>},
   {"aten::split_with_sizes_copy",  op::translate_split_with_sizes},
   ```

2. **Different signature but same underlying op** → write a thin wrapper that adapts inputs/attrs and dispatches to the existing translator.

3. **Genuinely new behavior** → implement a new translator under `src/frontends/pytorch/src/op/` and register it in both tables if the FX path also emits it.

Add an FX-table entry (`aten.<name>.default` / specific overload) only if the FX path also reports the op missing.

**Ensure the alias is actually covered by a TorchScript test.** Many existing `*_copy` layer tests are marked only `@pytest.mark.precommit_fx_backend` (and sometimes `precommit_torch_export`), so the TS table entry you just added is never exercised by the TorchScript precommit job. After adding a TS alias, find the matching test class (e.g. `TestSelectCopy`, `TestViewCopy`) and add `@pytest.mark.nightly` + `@pytest.mark.precommit` so the new registration is validated. If no test exists for the op, add one. Reviewers (including Copilot) flag missing TS coverage for new TS registrations.

### Bucket B — Translator only registered for FX

Symptom: TS test fails with "No translator found for `aten::<op>`" while an `*_fx` translator already exists.

Fix: register the existing `translate_<op>_fx` (or write a thin TS wrapper) in the TS section of `op_table.cpp`, and extend the translator body to accept the TS calling convention. Use `num_inputs_check(context, min, max)` and branch on input count for the TS dtype/layout/device tail. Example: see `translate_scalar_tensor_fx` in [src/frontends/pytorch/src/op/scalar_tensor.cpp](../../../src/frontends/pytorch/src/op/scalar_tensor.cpp).

As in Bucket A, add `@pytest.mark.nightly` + `@pytest.mark.precommit` to the corresponding test so the TS path is exercised. If you added a new TS-only code branch (e.g. a TS dtype tail), add a parametrized case that hits it — extending the existing test, not duplicating it.

### Bucket C — FX-only test code reached in TorchScript mode

This bucket only applies when a test ends up executed in the TS path while its body relies on FX-only constructs (`torch.cond`, `torch.while_loop`, `torch.ops.aten.*`, `torch.ops.quantized_decomposed.*`, etc.). It happens when:

1. The test class carries both `@pytest.mark.precommit` and `@pytest.mark.precommit_torch_export` but the body works only on one path, **or**
2. The suite is run without `-m` (e.g. ad-hoc full-suite reproduction).

If the test is genuinely FX-only, drop the `precommit` marker so `-m precommit` skips it cleanly. If both backends are intended, add a body-level guard:

```python
if not (self.use_torch_export() or self.use_torch_compile_backend()):
    pytest.skip("FX-only test")
```

For parametrize-time skipping, the helpers in [tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py](../../../tests/layer_tests/pytorch_tests/pytorch_layer_test_class.py) are `skip_if_export(*params, reason=...)` (skips on the `torch.export` path), `skip_if_fx(*params, reason=...)` (skips on the `torch.compile`/FX path), and `skip_check(*params, reason=...)` (skips on whichever non-TS path is active). There is **no** `skip_if_ts` helper — to skip the TorchScript path, gate inside the test body with `if not (self.use_torch_export() or self.use_torch_compile_backend()): pytest.skip(...)`.

### Bucket D — Stricter typing / changed default kwargs

Examples observed in past bumps:

- `aten::hardtanh_` is now emitted instead of `aten::hardtanh` when `inplace=True` — make the expected op-kind dynamic in the test.
- Literal `[0, 1.0]` rejected as mixed int/float — change to `[0.0, 1.0]`.
- New kwargs default-on (e.g. `scaled_dot_product_attention(enable_gqa=...)`) — pass the value explicitly so traced graphs match across versions.

### Bucket E — Accuracy regressions

Before assuming a translator bug, check whether the same input pattern triggers a known plugin issue (e.g. CPU `Multiply(x, Constant(inf))` followed by `IsInf` with dynamic shape returns all-False due to a Mul+IsInf fusion). If reproducible without the PT FE, work around in the test input and note the underlying ticket in the commit message — do not paper over real frontend regressions.

A fast way to confirm a failure is pre-existing rather than bump-induced: reinstall the **previous** torch/torchvision pins into a scratch virtualenv (`python3 -m venv --system-site-packages /tmp/prevtorch && /tmp/prevtorch/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==<old>.0`) and rerun just the failing test with that interpreter against the **same** OV build/commit. If it fails identically on both torch versions, the regression predates this bump (e.g. a genuine accuracy gap between the model's own reduced-precision path and the plugin's `inference_precision` hint) and doesn't need a code fix here — just note it in the commit message. If it only fails on the new torch version, it's bump-induced and needs a real fix.

## Step 4: Re-validate

Re-run both marker scopes from Step 2 until each ends with `0 failed`:

```bash
python3 -m pytest pytorch_tests/ -m precommit -n auto --tb=line -q 2>&1 | tail -5
PYTORCH_TRACING_MODE=EXPORT \
python3 -m pytest pytorch_tests/ -m precommit_torch_export -n auto --tb=line -q 2>&1 | tail -5
```

Expect a handful of new skipped/xfailed tests after Bucket C fixes — that is normal. FX-path failures usually point to missing FX-table entries or decomposition changes, not TS aliasing.

## Step 5: Check Model Hub Tests

Model hub tests load real (downloaded) models and prepare inputs using torchvision/torch helpers directly in the test file. These helpers can break when the new torchvision version removes or renames an API.

**Where to look**: `tests/model_hub_tests/pytorch/test_torchvision_models.py` and similar files in that directory. Focus on `load_model()` — it typically contains model-specific input preparation branches.

**Common failure pattern**:

```
ImportError: cannot import name '<name>' from 'torchvision.<module>'
```

A helper function inside the test file imports an API (e.g. `torchvision.io.read_video`, `torchvision.transforms.*`) that was removed in the new version. The function is called from inside a `load_model()` branch for a specific model (e.g. optical-flow models like RAFT).

**Fix**:

1. Identify what input shape the removed pipeline produced. Check the original code for resize/transform dimensions applied to the input.
2. Replace the helper call with a `torch.randn(...)` tensor of the same shape.
3. Remove the now-unused imports and the helper function itself.

Example (optical-flow model input after a 520×960 resize+transform pipeline):

```python
# before
frames = get_video()
self.example = prepare_frames_for_raft(model_name, [frames[100], frames[150]], ...)

# after
self.example = (torch.randn(2, 3, 520, 960), torch.randn(2, 3, 520, 960))
self.inputs  = (torch.randn(2, 3, 520, 960), torch.randn(2, 3, 520, 960))
```

Also check that any imports only needed by the removed helper are cleaned up (e.g. `import tempfile`, `import torchvision.transforms.functional as F`, `get_model_weights`).

**Verify** by checking the torchvision changelog for removed APIs whenever tests in this directory fail after a version bump.

## Step 6: Commit

One commit on a feature branch (project convention: `<user>/pt_fe/<short-topic>`, e.g. `mvafin/pt_fe/torch_2_12_upgrade`) covering: requirements, `op_table.cpp` additions, any translator extensions, and test fixes. List the new translators and the broken-test categories in the commit body — reviewers use it to map CI changes to user-visible behavior.

The pre-commit hook runs `clang-format` and may amend `src/frontends/pytorch/src/op/*.cpp`; re-stage and recommit if it does.

## Pitfalls

- **Model hub test helpers** (in `tests/model_hub_tests/pytorch/`) sometimes import torchvision/torch APIs that are removed in a new release. The failure surfaces as an `ImportError` inside a subprocess (via `multiprocessing_run`), so the traceback points into the helper, not the test parametrize line. Always check the torchvision changelog for removed APIs when these tests fail after a bump. Replace removed API calls with equivalent `torch.randn(...)` inputs of the same shape.
- **Do not** alias a new aten op to an existing translator without confirming semantic equivalence (signature, dtype/broadcast rules, attribute defaults). Aliasing is only safe when the new op is a behavioral no-op or true synonym of the existing one.
- **Do not** rely on `id(node)` when walking ov.Node graphs in Python; wrappers are recreated. Use `get_friendly_name()`.
- Test markers are filterable but not enforced — a class marked only `precommit_torch_export` still runs under TS if pytest is invoked without `-m`. Use marker-filtered invocations from Step 2; reserve inline `pytest.skip` for genuinely dual-marked tests.
- New TS op_table aliases/registrations are easy to leave untested: their tests are frequently marked `precommit_fx_backend` only. Always add `precommit` + `nightly` so the TorchScript job covers them (see Buckets A/B).
- `pytest -n auto` requires `pytest-xdist` (in `tests/requirements_pytorch`); confirm it is installed in the active env. Its importable module name is `xdist`, not `pytest_xdist` — `python3 -c "import pytest_xdist"` will falsely report it missing; check with `python3 -c "import xdist"` instead.
- `torchvision` minor = torch minor **+ 15** (`torch 2.9` ↔ `torchvision 0.24`, `torch 2.12` ↔ `torchvision 0.27`). Double-check the official release matrix before pinning.
- In `tests/requirements_pytorch`, place `--extra-index-url https://download.pytorch.org/whl/cpu` **before** the `torch==` and `torchvision==` lines so the PyTorch CPU index is available when pip resolves those packages. The line order matters for some pip-based tooling even though pip applies the option globally once parsed.
- `torchaudio` is intentionally **not** in `requirements_pytorch` (unused; dropped from PyTorch's official install since 2.8). Do not re-add it. If you ever must, note it is in a maintenance phase and lags torch (`torch 2.12.0` ↔ `torchaudio 2.11.0`), so pinning `torchaudio==<torch version>` blindly causes a CI install failure (`No matching distribution found for torchaudio==X.Y.0`) — look up the latest available wheel.

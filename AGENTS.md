# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core runtime, frontends, plugins, and language bindings (`src/bindings/python`, `src/plugins/*`).
- `tests/`: functional, stress, memory, time, fuzz, and component-specific test suites.
- `samples/`: C, C++, Python, and Node.js sample apps.
- `tools/`: developer and runtime tools (for example `benchmark_tool`, `ovc`).
- `docs/`: developer and user documentation, including platform build guides in `docs/dev/`.
- `cmake/` and top-level `CMakeLists.txt`: build system configuration.

## Build, Test, and Development Commands
- Install dependencies (Linux): `sudo ./install_build_dependencies.sh`
- Configure build: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build --parallel`
- Build with Python API: add `-DENABLE_PYTHON=ON` (and `-DENABLE_WHEEL=ON` for wheel packaging).
- Run all discovered tests: `ctest --test-dir build -V`
- Run component tests by label: `ctest --test-dir build -L CPU -V` (labels include `OV`, `CPU`, `GPU`, `ONNX_FE`, etc.).
- Generate coverage (when configured with `-DENABLE_COVERAGE=ON`): `make -C build ov_coverage`

## Coding Style & Naming Conventions
- C/C++ style uses `clang-format-18` with rules from `src/.clang-format`.
- Preferred API naming: classes in `PascalCase`; methods/functions in `snake_case` (see `docs/dev/coding_style.md`).
- Naming checks can be enabled with `-DENABLE_NCC_STYLE=ON`; formatting fixes via `clang_format_fix_all` target.
- Python formatting is configured in `pyproject.toml` (`black`, line length `160`).

## Testing Guidelines
- Keep tests near the affected component under `tests/` or component-local test folders in `src/plugins/*/tests`.
- Name and group tests so they can run through CTest labels and existing test binaries (for example `ieUnitTests`, `ieFuncTests`).
- Validate locally before PR creation by running relevant component tests, not only full-suite CI.
- Coverage workflow (`.github/workflows/coverage.yml`) now runs a consolidated list of binaries and continues on per-test failures to still generate/upload `coverage.info` and HTML artifacts.
- Memory smoke binaries are included in coverage: `test_inference_async` and `test_inference_sync` (using `src/core/tests/models/ir/add_abc.xml`).

## Coverage Workflow Notes
- Coverage job explicitly enables test/frontends flags: `-DENABLE_FUNCTIONAL_TESTS=ON`, `-DENABLE_OV_PADDLE_FRONTEND=ON`, `-DENABLE_OV_TF_FRONTEND=ON`, `-DENABLE_OV_TF_LITE_FRONTEND=ON`.
- Additional dependency installed for coverage tests: `src/frontends/tensorflow_lite/tests/requirements.txt`.
- `paddle_test` requests map to the `paddle_tests` binary name in this repository.
- Known limitations in `coverage.yml`:
  - `ov_npu_func_tests` and `ov_npu_unit_tests` are skipped (NPU-specific environment/driver constraints).
  - `ov_nvidia_func_tests` is skipped (requires `openvino_contrib` NVIDIA plugin build and NVIDIA-enabled runner).
- If you add new heavy tests, prefer smoke filters first, then widen scope after confirming runtime and stability.

## Commit & Pull Request Guidelines
- Follow recent commit style: optional scope tags plus imperative subject, often with PR reference, e.g. `[Core] Fix X (#12345)`.
- Keep PRs focused on one issue, link the related issue, and use Draft PRs for WIP.
- Rebase/merge latest target branch before opening PR.
- Ensure CI/pre-commit checks are green and include docs updates when user-facing behavior changes.

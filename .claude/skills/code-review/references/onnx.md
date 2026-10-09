# ONNX Frontend Review

Apply to changes under `src/frontends/onnx/`, in addition to the
[frontend review guidance](frontend.md).

## Tests

- C++ conversion tests in `src/frontends/onnx/tests/*.in.cpp` build an
  `ov::test::TestCase`. A test must call `run()` or `run_with_tolerance_as_fp()`;
  otherwise it neither infers nor compares.
- `run()` compares by mantissa bits, so it is too strict for accumulating
  float ops. `run_with_tolerance_as_fp()` uses an absolute tolerance, so check
  it against the output magnitude. `TestCase` compiles models with f32
  inference precision on every device.
- When conversion support changes, check and update the expected-failure
  lists:
  - strict xfails in `src/frontends/onnx/tests/tests_python/test_backend.py`,
    together with the reason strings in `src/frontends/onnx/tests/__init__.py`;
  - `src/frontends/onnx/tests/unit_test.manifest` and the per-backend manifests
    under `src/frontends/onnx/tests/runtime/*/unit_test.manifest`.
- Check that test models (`src/frontends/onnx/tests/models/*.prototxt`) are valid ONNX for the
  declared opset: input element types match the operator's type constraints,
  scalar inputs have no `dims`, and operators exist in the imported opset.
  Prefer cross-checking expected values with onnxruntime or the ONNX reference
  implementation.

# OpenVINO Frontend Review

Apply to ONNX, TensorFlow, TFLite, PyTorch, JAX, Paddle, GGUF, and other
frontend changes.

## Review focus

- Keep frontend code hardware- and plugin-agnostic. Conversion must preserve
  framework semantics without targeting a specific device or plugin. Flag
  device-dependent conversion branches or backend-specific optimizations;
  those belong in the relevant plugin's transformation or execution pipeline.
- Check rank, dimension, element type, layout, optional inputs, attributes,
  opset or framework-version behavior, and dynamic-shape handling.
- Verify that conversion emits the intended OpenVINO graph and preserves
  framework semantics for valid inputs.
- Check malformed, missing, out-of-range, duplicated, and unsupported values
  for deterministic validation and useful errors.
- Check registration tables, opset ranges, translator dispatch, normalization
  passes, and framework-specific entry points for consistency.
- Check that conversion does not depend on unavailable compile-time shape or
  type information when runtime graph computation is required.
- Check that user-visible frontend behavior has focused regression tests,
  including dynamic, boundary, and malformed-input cases where applicable.

## Evidence and scope

Do not report unsupported framework behavior unless the diff claims to
support it or changes its error handling. Check that fallback nodes or
placeholder translators are not presented as full support.

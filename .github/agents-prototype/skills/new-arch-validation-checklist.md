# New Architecture Validation Checklist

This checklist is for validating enablement of models that introduce architecturally
novel patterns. Run through every category before submitting a patch or opening a PR.

## How to Use

For each section, mark each item as `PASS`, `FAIL`, or `N/A`.

A PR must have **no FAIL items** before it can be merged.

---

## 1. FP16 / BF16 Precision

- [ ] **Integer indexing paths** — position IDs, sequence indices, attention mask indices are
  kept in `int32` or `int64` throughout; no cast to FP16/BF16 on index tensors.
- [ ] **Scale factors** — any scalar multiplier absorbed into a weight matrix via
  `MatMulMultiplyFusion` has `|scalar| ≤ 1.0` OR the weight dtype is FP32. If `|scalar| > 1.0`
  AND weight dtype is FP16/BF16 — the fusion must be skipped for that instance.
- [ ] **Constant tables** — cached trig tables (`cos_cached`, `sin_cached`), embedding lookup
  tables, and other large constant tensors are stored at the precision required by the op spec,
  not silently downcast during constant folding.
- [ ] **Reference comparison** — model inference result with `FP32` and `FP16` differ by no
  more than the expected numerical tolerance (≤ 1% max absolute difference on representative
  inputs, or per model card specification).

## 2. Shared KV Cache

- [ ] **ReadValue fan-out** — for every `ReadValue` node that is part of a KV cache:
  `readvalue->get_output_target_inputs(0).size() == 1`. If `> 1`, the state is shared and
  any fusion or folding pass that assumes 1-to-1 state ownership MUST be skipped.
- [ ] **Assign pairing** — each `ReadValue` has exactly one corresponding `Assign` that writes
  to the same state variable. If `> 1` Assigns write to the same state — this is a model
  defect; file a bug before proceeding.
- [ ] **StatefulSDPAFusion guard** — the `StatefulSDPAFusion` pass is guarded against shared KV
  cache; a shared `ReadValue` causes `return false` from the callback, not a fusion.
- [ ] **NPUW fold idempotency** — any weight-folding pass that touches state-connected nodes
  is tagged (e.g. `FoldedTag`) after the first application and skipped on re-application.

## 3. Novel Tensor Types

- [ ] **ONNX SequenceMark invariant** — every op that produces a sequence output wraps it in
  `ov::frontend::SequenceMark` at the point of creation in the translator.
  No downstream transformation receives a bare sequence node without `SequenceMark`.
- [ ] **Mark operations** — `ComplexTypeMark`, `SequenceMark`, and similar wrappers are used
  for all special tensor types; the normalize-step transformation resolves them before
  plugin execution.
- [ ] **per_layer_inputs** — genuinely novel tensor types that cannot be expressed via existing
  OV ops are registered as `per_layer_inputs` in NPUW partition metadata; they are NOT handled
  by adding special cases to the main inference path.

## 4. IR Correctness

- [ ] **Op spec compliance** — every new op's output shapes and types match the published op spec
  in `docs/articles_en/documentation/openvino-ir-format/operation-sets/`.
- [ ] **No `PtFrameworkNode` in converted graph** — the final OV IR contains no unresolved
  framework nodes (ONNX: no `ONNXFrameworkNode`; PyTorch: no `PtFrameworkNode`).
- [ ] **clang-format** — all changed `.hpp` / `.cpp` files pass `clang-format` with
  `src/.clang-format` config.
- [ ] **clang-tidy** — no new warnings or errors introduced in changed files.

## 5. Test Coverage

- [ ] **Positive test** — a test that exercises the exact new op/transformation and verifies
  the expected output node appears in the resulting graph.
- [ ] **Negative test** — a test that exercises a non-matching input graph and verifies the
  transformation does NOT fire (graph unchanged).
- [ ] **Nightly coverage** — new op translators have at least one test marked for nightly/regression
  mode, not only `@pytest.mark.precommit`.
- [ ] **Accuracy regression test** — if the fix addresses an accuracy issue, a test that
  directly reproduces the failing scenario is included in the patch.
- [ ] **No disabled tests** — no tests are skipped or disabled without a documented reason
  (`pytest.mark.skip(reason=...)` or `GTEST_SKIP()` with a GitHub issue reference).

# Example Diagnosis Report — MoE3GeMMFusion (transpose_b mismatch)

This is a filled example of the output the skill produces. Use it as a quality reference when writing your own diagnosis report.

---

## MatcherPass Diagnosis: MoE3GeMMFusion

**Root cause:** All three `MatMul` nodes (gate, up, down projections) in the model's MoE subgraph have `transpose_b=false`, but the `MoE3GeMMFusion` pattern hard-requires `transpose_b=true` via an `attrs_match` predicate. No explicit `Transpose` node wraps the weight input either, so neither of the pattern's two matching paths can succeed.

**Log evidence:**
```
{  MATCHING PATTERN NODE: WrapType<MatMul>(WrapType, any_input)
├─ AGAINST  GRAPH   NODE: MatMul(Multiply, Reshape)
├─ PREDICATE `attrs_match({ transpose_b: YES, transpose_a: NO }) && consumers_count(1)` FAILED
}  NODES' TYPE MATCHED, but PREDICATE FAILED
```
This phrase appears **30 times** in the matcher log — once per MoE layer candidate — confirming all 30 match attempts fail at exactly the same point.

**Failing node:** `ov::op::v0::MatMul` (down-projection, and symmetrically gate/up-projection MatMuls). Op type is correct; only the `transpose_b` attribute fails the predicate.

**Resolution:**
- File: `src/plugins/intel_cpu/src/transformations/cpu_opset/common/pass/moe_matmuls_fusion.cpp`, lines ~241, 248, 253
- Remove the `{"transpose_b", true}` constraint from all three `wrap_type<MatMul>` predicate maps and instead read the actual flag inside the callback via `gate_mm_node->get_transpose_b()`, then propagate it to `BatchGatherMatmul`.
- Alternative: add a second pattern branch (via `pattern::op::Or`) covering `transpose_b=false`.

## Reproducer Test
File: `src/plugins/intel_cpu/tests/unit/transformations/moe_matmuls_fusion_test.cpp`
Test name: `MoE3GeMMFusion_TransposeBFalse_TestF.MoE3GeMMFusion_TransposeBFalse_NotApplied`
Status before fix: **PASS (green)** — transformation does not fire; model unchanged equals the auto-cloned `model_ref`. Matcher log from the test run shows the identical `attrs_match({ transpose_b: YES })` predicate failure as the original model, confirming the reproducer is faithful.

---

## What makes this report good

- **Log evidence is a direct quote** from the actual `OV_MATCHER_LOGGING` output — not paraphrased from source code inspection.
- **The count (30)** is confirmed from the log, not inferred from the model structure.
- **Resolution points to exact lines**, not just a filename.
- **Reproducer test status is "PASS (green)"** with an explanation of why — the skill's harness auto-clones `model` into `model_ref`, so no transformation firing = models equal = green. A red test before the fix would mean the test was written incorrectly.
- **Cross-validation is stated** — the test's matcher log matches the original model's log phrase.

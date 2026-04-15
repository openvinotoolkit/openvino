# Skill: OpenVINO Transformation Analysis

## Purpose

Analyse an OpenVINO IR sub-graph to identify the optimal fusion pattern and
transformation type before any implementation work begins. Produces a structured
`transformation_analysis.md` that the implementation skill consumes.

## When to invoke

- Transformation Agent Step 1
- Any time you need to decide *which* `ov::pass` class to use and *why*

---

## Steps

### Step 1: Load Op Spec and Existing Graph

Obtain the op spec from agent-results:
```bash
# Op spec is written by Core OpSpec Agent to agent-results/core-opspec/
OP_SPEC_PATH=$(python3 -c "
import json
try:
    d = json.load(open('agent-results/core-opspec/core_opspec_result.json'))
    print(d.get('op_spec_path', ''))
except Exception:
    print('')
")
[ -n "$OP_SPEC_PATH" ] && cat "$OP_SPEC_PATH" || echo "[WARN] Op spec not found in agent-results/core-opspec/"
```

Export the model to IR (using cached IR if available) and visualise the target
sub-graph:
```python
import openvino as ov

model = ov.Core().read_model("openvino_model.xml")
# Find the target nodes by op type or friendly name
target = [n for n in model.get_ordered_ops()
          if n.get_type_name() in {"MatMul", "Add", "Tanh"}]
for n in target:
    print(n.get_friendly_name(), n.get_type_name(),
          [i.get_shape() for i in n.inputs()])
```

### Step 2: Draw the Sub-Graph Pattern

Produce a textual node diagram:
```
Parameter(input) ──► MatMul ──────────────► Add ──► Result
                       ▲                     ▲
Constant(weights) ─────┘    Constant(bias) ──┘
```

For each node record:
- Op type and version (e.g. `MatMul v0`)
- Input sources: parameter (dynamic), constant (static), intermediate (op output)
- Output consumers (does anything else use this output?)
- Shape and dtype at each port

### Step 3: Identify Constant Constraints

A `MatcherPass` can only reliably fuse when certain inputs are `Constant` nodes
(otherwise the fusion is not safe at compile time). List each input and whether
it must be constant for the fusion to be valid.

If non-constant inputs must be treated as runtime values, the transformation may
need to be a `FunctionPass` or use symbolic shape analysis.

### Step 4: Classify Transformation Type

Apply this decision tree:

```
Is the fusion local (fixed sub-graph topology)?
  YES → Does it need to run before any other pass touches these nodes?
    YES → MatcherPass registered early in common_optimizations
    NO  → MatcherPass registered at the appropriate priority slot
  NO → Does the pass need consumer information (backward)?
    YES → BackwardGraphRewrite
    NO  → FunctionPass (iterate over all nodes manually)
```

| Type | When | Examples in codebase |
|---|---|---|
| `MatcherPass` | Local topology, pattern of 2-10 ops | `FuseMHATokensSplit`, `FuseConvolutionSimpleLeakyRelu` |
| `FunctionPass` | Multi-pattern or graph-wide | `ConstantFolding`, `AlignMixedFP32FP16Types` |
| `BackwardGraphRewrite` | Consumer-first needed | `ConvertPrecision` (when checking all consumers) |
| `GraphRewrite` | Forward multi-pattern | Wraps multiple `MatcherPass` instances |

### Step 5: Find a Template to Follow

Search for the syntactically closest existing transformation:
```bash
# Find transformations that match a similar pattern (e.g., MatMul + Add fusion)
grep -r "MatMul\|GemmFusion\|LinearFusion" \
  src/common/transformations/include/transformations/ --include="*.hpp" -l
```

Read that file fully — it is your implementation template.

### Step 6: Output `transformation_analysis.md`

```markdown
## Transformation Analysis

**Target pattern:**
```
<ASCII diagram>
```

**Transformation type:** MatcherPass

**Rationale:** The pattern has a fixed topology of 3 ops
(MatMul, Add, optional Tanh) with constant weights and bias.
MatcherPass is optimal — it fires once per match with O(n) complexity.

**Constant constraints:**
- weights: MUST be Constant (used as a compile-time weight tensor)
- bias: MUST be Constant

**Template reference:**
`src/common/transformations/src/transformations/common_optimizations/fuse_u4_weights_zero_point.cpp`

**Registration target:**
`src/common/transformations/src/transformations/common_optimizations/common_optimizations.cpp`
via `ADD_MATCHER(manager, FuseMyLinearFusion)`
```

---

## Output

`transformation_analysis.md` written to `artifacts/` or posted as issue comment.
Contains everything required by `openvino_transformation_implementation` to start
coding without further research.

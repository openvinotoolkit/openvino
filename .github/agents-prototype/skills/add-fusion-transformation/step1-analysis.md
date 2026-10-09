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

Obtain the op spec from `agent-results/core-opspec/` (written by Core OpSpec Agent).
If not available, find the op specification in the OpenVINO documentation directory:
`docs/articles_en/documentation/openvino-ir-format/operation-sets/`

Export the target model to IR and locate the sub-graph visually using the
OpenVINO Model Explorer or by reading the XML file directly. Focus on
identifying:
- Which node types appear before/after the target op
- Constant vs. dynamic inputs on each node
- Shape and dtype at each port
- Output consumer count for each node

### Step 2: Draw the Sub-Graph Pattern

Produce a textual node diagram (**this is an example structure — adapt to your actual sub-graph**):

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

Search for the syntactically closest existing transformation in `src/common/transformations/`
by subgraph depth and operator types. For example, if your pattern fuses two ops, look for
existing two-op fusions; if it decomposes a complex op, look for similar decompositions.

Search the headers directory to find candidates:

```
dir src\common\transformations\include\transformations\common_optimizations\
```

Read the closest match in full — it is your implementation template.
Do not copy-paste; understand each section before adapting it to your pattern.

### Step 6: Output `transformation_analysis.md`

Write your analysis result to `agent-results/transformation/transformation_analysis.md`.
Include at minimum:
- ASCII diagram of the matched sub-graph
- Chosen transformation type and rationale
- Constant constraints for each input
- Template reference (path to the transformation used as template)
- Registration target (which pipeline file and position)

**Example output structure** (adapt content to your specific transformation):

```
## Transformation Analysis

Target pattern: [ASCII diagram]

Transformation type: MatcherPass

Rationale: Fixed topology of 3 ops with constant weights and bias.
MatcherPass is optimal — fires once per match with O(n) complexity.

Constant constraints:
- weights: MUST be Constant
- bias: MUST be Constant

Template reference:
src/common/transformations/src/transformations/common_optimizations/fuse_u4_weights_zero_point.cpp

Registration target:
src/common/transformations/src/transformations/common_optimizations/common_optimizations.cpp
```

---

## Output

`transformation_analysis.md` written to `artifacts/` or posted as issue comment.
Contains everything required by `openvino_transformation_implementation` to start
coding without further research.

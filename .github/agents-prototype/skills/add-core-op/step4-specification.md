# Skill: Core Op Specification

> Source: `skills/add-core-op/SKILL.md` (Step 4)
> Agent: `core_opspec_agent`

## Prerequisites

- Completed **core_op_testing** skill - op is implemented and all tests pass.

## Specification File

**Location:**
`openvino/docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/<category>/<op_name>-<opset_version>.rst`

**Example:**
`openvino/docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/signals/istft-16.rst`

## Categories

Place the `.rst` file in the appropriate category subdirectory:

| Category | Operations |
|----------|-----------|
| `arithmetic` | Add, Subtract, Multiply, etc. |
| `comparison` | Equal, Greater, Less, etc. |
| `convolution` | Convolution, GroupConvolution, etc. |
| `detection` | DetectionOutput, NMS, etc. |
| `image` | Interpolate, etc. |
| `infrastructure` | Constant, Parameter, Result, etc. |
| `logical` | LogicalAnd, LogicalOr, etc. |
| `matrix` | MatMul, Einsum, etc. |
| `movement` | Gather, Scatter, Reshape, etc. |
| `normalization` | BatchNorm, LRN, MVN, etc. |
| `pooling` | MaxPool, AvgPool, etc. |
| `quantization` | FakeQuantize, etc. |
| `reduction` | ReduceMax, ReduceSum, etc. |
| `sequence` | LSTMSequence, GRUSequence, etc. |
| `signals` | DFT, IDFT, STFT, ISTFT, etc. |
| `sort` | TopK, NonMaxSuppression, etc. |
| `type` | Convert, ConvertLike, etc. |

## RST Structure

Follow the existing spec conventions. Typical structure:

```rst
.. meta::
   :description: Learn about the OpName-X operation.

.. _openvino_docs_ops_<category>_OpName_X:

OpName - opset X
================

.. csv-table::
   :header: "Attribute", "Description"

   "Version", "opset X"
   "Category", "<category>"
   "Brief", "<one-line description>"

Description
-----------

<Detailed description of the operation, math formula, semantics>

Attributes
----------

<Table of attributes with name, type, default, description>

Inputs
------

<Table of inputs with index, name, type, description>

Outputs
-------

<Table of outputs with index, name, type, description>

Types
-----

<Supported element types>

Detailed Description
--------------------

<Math formula, broadcasting rules, edge cases>

Examples
--------

<XML/IR examples showing the op in a model>
```

## Validation

- Spec covers all inputs, outputs, and attributes from the implementation.
- Math formula matches the `evaluate()` reference kernel.
- Supported types match `has_evaluate()`.
- Category matches the operation's purpose.

## Output

- Completed `.rst` specification file.
- All 4 steps done → report `success` + branch/patch to OV Orchestrator.

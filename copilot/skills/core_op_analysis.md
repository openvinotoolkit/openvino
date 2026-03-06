# Skill: Core Op Analysis

> Source: `.github/skills/add-core-op/SKILL.md` (Step 1)
> Agent: `core_opspec_agent`

## When to Use

- A model conversion fails with `No conversion rule for <OpName>`.
- An operation is missing from OpenVINO's op set.
- Decomposition of the missing op is not possible or not performant enough.
- No existing operator can implement the required functionality.

## Procedure

1. **Identify the missing operation** from the error context / conversion log.
   - Extract the exact op name (e.g. ONNX name, PyTorch ATen name, TF op name).
   - Determine the source framework (ONNX, PyTorch, TensorFlow).

2. **Research the math formula / semantics:**
   - Official specification (ONNX spec, PyTorch docs, TF docs).
   - Inputs, outputs, attributes, data types, broadcasting rules.

3. **Check alignment with OpenVINO frontends:**
   - Does PyTorch FE / ONNX FE / TF FE already have a mapping that decomposes
     this op into existing OpenVINO ops?
   - If yes → decomposition may be sufficient (no new core op needed).
   - If no or decomposition is too slow → new core op is needed.

4. **Check existing OpenVINO opsets:**
   - Is there a similar op in a previous opset that can be extended?
   - If adding a new version of an existing op, note the version delta.

5. **Collect references:**
   - Link to the op specification in the source framework.
   - Link to any related OpenVINO issues or PRs.
   - Link to the model that requires this op (`model_id`).

6. **Determine the target opset:**
   - New ops are registered only in the **latest opset** (e.g. `opset16`).
   - Use the latest opset version number for the namespace `ov::op::vX`.

## Output

Return a structured analysis:

```
op_name:        <OpName>
source_framework: <ONNX|PyTorch|TF>
math_formula:   <brief description>
inputs:         <list of inputs with types>
outputs:        <list of outputs with types>
attributes:     <list of attributes>
target_opset:   <opsetX>
decomposable:   <yes|no>
reason:         <why new core op is needed>
references:     <list of URLs>
```

If `decomposable=yes` - report back to OV Orchestrator; no further core op
work needed (defer to Transformation agent for decomposition).

If `decomposable=no` - proceed to **core_op_implementation** skill.

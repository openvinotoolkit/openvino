---
name: dev-nope-elimination-transformation
description: Develop a universal OpenVINO nope-elimination transformation that finds user-specified nodes and replaces or removes them to minimize the graph and improve performance, with correctness checks and tests.
---

# Development Skill: Nope Elimination Transformation (Find/Replace/Remove)

## Purpose
Create or update a transformation that locates specified node types or patterns and replaces or removes them to minimize the graph and improve performance, while preserving mathematical correctness. Include detection of no-op nodes and redundant sequences (for example, reshape-to-same-shape or double-reshape that returns to the original shape).

## Inputs to Collect
- Target node types or pattern description (op types, opset versions, attributes, inputs/outputs).
- Replace or remove behavior and desired replacement nodes.
- Known constraints (shape, dtype, layout, axis semantics, broadcast rules).
- No-op or redundant shape patterns to target (for example, reshape-to-same-shape or reshape chain that returns to the original shape).
- Target component (core transformations, plugin-specific, frontend).
- Test assets or minimal graphs to reproduce.

## Guardrails
- Only apply the transformation when correctness can be guaranteed.
- If correctness cannot be proven for a pattern, skip the rewrite.
- Avoid changing observable behavior (numerical results, output shapes, dynamic ranks).
- Prefer early exits with clear conditions rather than speculative rewrites.

## Steps
1. Check [src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp](src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp) for a similar transformation; extend it if possible, otherwise create a new transformation in that file.
2. Read the existing transformation pipeline and related passes to find the correct pass manager and insertion point.
3. Define the pattern for the user-specified nodes with explicit opset versions and attribute checks.
4. Add explicit patterns for no-op nodes and redundant sequences (for example, reshape-to-same-shape or reshape chain that returns to the original shape).
5. Identify all preconditions needed for correctness (shape/rank, constants, value ranges, monotonicity, etc.).
6. Implement the replacement or removal, preserving friendly names and runtime info.
7. Add explicit checks to skip the transformation when any precondition fails.
8. Add unit tests to verify graph rewriting (match count, replaced nodes, and structural properties).
9. Add functional tests to verify inference matches baseline outputs.
10. Document edge cases and why they are skipped (if applicable). Provide a simple examples in the transformation description

## Implementation Notes
- Use `ov::replace_node` and `ov::copy_runtime_info` to preserve metadata.
- The callback function should be defined directly inside the transformation..
- Keep changes minimal and avoid duplicating logic from existing utilities. Use existing helper functions, extend them, or create new ones for common helpers (src/common/transformations/src/transformations/utils). Keep transformation-specific helpers within the transformation (Prefer using lambda functions).
- If removing nodes, ensure output consumers are redirected safely.
- If replacing, keep output element types and shapes consistent.
- For reshape elimination, prove that output shape is identical to input shape or that a reshape chain returns to the original shape.

## Test Expectations
- Unit tests: graph-level assertions that the transformation modified the graph correctly (Test structure by default: generate a test model, apply the transformation, and compare the transformed model with the reference model)
- Functional tests: end-to-end inference comparison before and after transformation.
- Add a negative test that confirms the pass is skipped when preconditions are not met.
- Add tests covering no-op reshape and redundant reshape chains.

## Success Criteria
- Transformation fires only when safe.
- Unit tests prove the graph is modified correctly.
- Functional tests prove inference results match the original model within tolerance.

## Example Usage
Use the `dev-nope-elimination-transformation` skill when the request is:
"Develop a new transformation to eliminate no-op and redundant Reshape nodes. Start by checking and extending [src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp](src/common/transformations/src/transformations/common_optimizations/nop_elimination.cpp). Add unit tests to verify graph rewrites and functional tests to verify inference equivalence. Skip dynamic-shape cases or any pattern that cannot be proven correct."

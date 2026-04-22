---
name: add-fe-op
description: Adds a new operation to OpenVINO Frontend pipelines with translator updates, registration, and tests.
---

# Agent Skill: Add New Operation to OpenVINO Frontend

## Goal
Provide a clear instruction-only workflow to enable a new operation in the OpenVINO Frontend (FE) pipeline.

Expected outcome:

- Operation translation
- Conversion to OpenVINO graph
- Unit and functional validation

---

## Scope

This instruction applies to these frontends:

- `pytorch`
- `tf` (TensorFlow)
- `onnx`

The workflow covers:

1. Translator creation/update
2. Translator registration in op mapping tables
3. Conversion validation
4. Test updates

---

## Architecture

Agent → Skill → FE Translator → OpenVINO Core Graph

---

## Expected Repository Layout

openvino/
├─ src/frontends/<framework>/
│  ├─ src/
│  │  ├─ op/
│  │  │  └─ <new_op>.cpp
│  │  ├─ op_table.cpp
│  │  └─ CMakeLists.txt
│  └─ tests/
│     └─ test_<new_op>.cpp
├─ tests/
│  └─ layer_tests/
│     └─ <framework>/
│        └─ test_<new_op>_conversion.py

---

## Instruction Workflow

### 1) Check current support state
- Verify whether translator file already exists.
- Verify whether op is already registered in frontend mapping table.
- If both exist and conversion is known to pass, skip scaffolding.
- If only partial state exists, repair only missing parts.

### 2) Add translator logic
- Preferred path: emit real OV conversion logic when the operation maps to an OpenVINO op or a set of operations that will provide reasonable performance.
- If the same operation already exists in another frontend, extract/reuse common translation logic instead of duplicating it.
- For simple 1:1 ops, prefer existing helper translation patterns/utilities already used in FE codebase.
- Fallback path: emit safe placeholder translator only when real mapping is unavailable.
- Never claim full support when fallback translator is used.

### 3) Register operation in frontend tables
- PyTorch:
    - Add converter declaration in op table.
    - Add TorchScript key registration (for example `aten::op`).
    - Add FX key registration (for example `aten.op.default`).
- TensorFlow:
    - For supported unary ops, prefer generic unary registration path.
    - For unsupported cases, register dedicated translator function.
- ONNX:
    - Ensure `ONNX_OP` macro registration exists in translator file.

### 4) Add tests
- Add frontend smoke test file under `tests/frontend` (not available for PyTorch). 
- Add framework layer test under `tests/layer_tests` where supported.
- Ensure test naming follows existing suite conventions.

### 5) Build prerequisites for validation
- Ensure frontend changes are available to runtime before conversion checks.
- Build frontend targets from source or use an existing OpenVINO build/package that already includes your FE changes.
- If neither is true, do not report conversion status as passed; mark validation as blocked.

### 6) Validate conversion and finalize
- run conversion check from generated <frontend> model.
- if automated validation is unavailable, mark as skipped with explicit reason.
- Confirm no duplicate registrations are introduced.
- Confirm rerun is idempotent.
- Report one of three states: fully enabled, partially repaired, or scaffolded with fallback.

## Translation Recommendations

- Prefer runtime shape computation over conversion-time shape extraction.
    - Use `ShapeOf`-based graph logic for shape-dependent behavior.
    - Avoid relying on `get_shape()` / fully static shape reads during FE conversion.
    - `get_partial_shape()` is acceptable for compile-time rank-only decisions.

- Handle data types according to framework semantics.
    - If PyTorch op behavior allows mixed input types, preserve this in translation.
    - Do not over-constrain translators to a single dtype when mixed-type execution is valid.

- Avoid constants tied to compile-time element type queries.
    - Do not create `Constant` values from `get_element_type()` when type can be dynamic at conversion time.
    - Prefer runtime type-safe construction paths that remain valid for dynamic element types.

## Notes

- Keep this skill instruction-only in markdown.
- Prefer minimal, root-cause updates over broad refactors.
- Do not mark operation as supported unless FE conversion produces OV graph nodes (not framework fallback nodes).

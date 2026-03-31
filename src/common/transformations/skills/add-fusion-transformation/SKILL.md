---
name: add-fusion-transformation
description: Adds a new OpenVINO fusion transformation (subgraph to one or several operations) and corresponding tests.
---

# Agent Skill: Add New Fusion Transformation in OpenVINO

## 🎯 Goal
Provide an instruction-only workflow to implement a new OpenVINO graph fusion transformation that replaces a matched subgraph with one operation (or a compact operation sequence), including regression tests.

Expected outcome:

- New transformation pass implementation
- Pass registration in the appropriate pipeline
- Positive and negative transformation tests
- Functional coverage when behavior is user-visible

---

## 🧩 Scope

This workflow applies to OpenVINO transformation development in `openvino/` for:

- Pattern-based graph fusion
- Subgraph replacement with:
  - a single fused op, or
  - several ops preserving semantics with lower runtime overhead

The workflow covers:

1. Pattern design and safety guards
2. Pass implementation and registration
3. Runtime correctness constraints (shape/type/friendly name/rt_info)
4. Unit and functional regression tests

---

## 🏗️ Architecture

Model Graph → Pattern Matcher Pass → Subgraph Replacement → Validation/Inference Tests

---

## 📂 Expected Repository Layout

openvino/
├─ src/common/transformations/
│  ├─ include/transformations/
│  │  └─ <domain>/<fusion_name>.hpp
│  ├─ src/transformations/
│  │  └─ <domain>/<fusion_name>.cpp
│  ├─ src/transformations/common_optimizations/
│  │  └─ common_optimizations.cpp   (or relevant pipeline file)
│  └─ tests/
│     ├─ common_optimizations/
│     │  └─ <fusion_name>.cpp
│     └─ utils/
└─ tests/
   └─ layer_tests/ or functional/    (if user-visible behavior/perf contract is affected)

---

## ✅ Instruction Workflow

### 1) Confirm the fusion does not already exist
- Search for equivalent matcher patterns and passes.
- If similar fusion exists, extend it instead of creating a duplicate pass.
- Reuse existing helper utilities and pattern predicates where possible.

### 2) Define fusion contract before coding
- Specify exact matched subgraph topology and constraints.
- Specify replacement graph (single op or multi-op compact form).
- Document semantic invariants:
  - output values
  - element types
  - ranks/dynamic-shape compatibility
  - runtime info / friendly names
- Identify explicit no-fuse conditions (negative guards).

### 3) Implement matcher pass
- Add pass class declaration in header and implementation in source.
- Prefer `ov::pass::MatcherPass` for local rewrite; use `GraphRewrite` only when grouping multiple related matchers.
- Make predicates strict enough to avoid unsafe rewrites.
- Preserve node names and `rt_info` where expected.
- Register and copy runtime info for replacement nodes.
- Avoid hidden behavior changes; if assumptions are violated, do not fuse.

### 4) Register pass in the right optimization pipeline
- Add the pass to the appropriate transformation pipeline file.
- Keep registration order consistent with dependencies (producer fusions before consumer fusions when required).
- Do not widen pass scope beyond the intended domain.

### 5) Add transformation unit tests
- Add tests under transformations test suite with:
  - positive case(s): pattern is fused
  - negative case(s): pattern must remain unchanged when guards fail
  - dynamic-shape/type coverage where relevant
- Compare transformed function against expected reference graph.
- Ensure tests validate both structure and semantics.

### 6) Add functional/layer tests when externally observable
- If fusion changes user-visible behavior/performance-sensitive execution path, add/extend functional tests.
- Cover representative input shapes/dtypes and backend-relevant conditions.
- Keep tests minimal and deterministic.

### 7) Validate and finalize
- Ensure pass is deterministic and idempotent.
- Verify no duplicated matcher registration.
- Confirm no fallback behavior silently changes model semantics.
- Report one of:
  - fully implemented + covered,
  - partially implemented (missing functional coverage),
  - blocked with explicit reason.

---

## 🧠 Fusion Design Recommendations

- Prefer root-cause optimization: fuse only when the resulting graph is strictly safer/faster or equivalent.
- Favor existing OpenVINO ops over introducing new custom ops unless required.
- Keep pattern predicates robust for partial/dynamic shapes.
- Avoid expensive checks in hot transformation loops when cheaper predicates exist.
- Reuse canonical helper utilities for constants, shape checks, and node replacement.

---

## ⚠️ Notes

- Keep this skill instruction-only in markdown.
- Do not claim fusion support without both positive and negative tests.
- Avoid broad refactoring unrelated to the fusion objective.
- Preserve compatibility with existing passes and test baselines.

```
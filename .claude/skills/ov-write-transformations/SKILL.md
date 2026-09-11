---
name: ov-write-transformations
description: >
  Write or review OpenVINO ov::Model graph transformations (passes: MatcherPass,
  ModelPass, GraphRewrite, BackwardGraphRewrite). Use when the user asks to write,
  add, or refactor a transformation/fusion/decomposition pass, choose between
  MatcherPass and ModelPass, write pattern-matching predicates, or bring an
  existing pass in line with OpenVINO transformation conventions.
---

1. Read [src/common/transformations/docs/writing_transformations.md](../../../src/common/transformations/docs/writing_transformations.md) — it covers where transformations live, choosing the pass type, anatomy of a MatcherPass, pattern matching and predicates, writing the callback, modifying the graph, documenting the pass, and what NOT to do.
2. Write or modify the pass following the patterns in that guide.
3. Add tests using the `ov-transformation-tests` skill, then build and run them to ensure the pass compiles and behaves as expected.

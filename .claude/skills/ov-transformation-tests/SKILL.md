---
name: ov-transformation-tests
description: >
  Write unit tests for OpenVINO ov::Model graph transformations (passes).
  Use when the user asks to write, add, or refactor tests for a transformation pass
  (MatcherPass, ModelPass), or to modernize legacy transformation tests.
---

1. Read `src/common/transformations/docs/writing_tests.md` — it covers test location, build target, fixture setup, FunctionsComparator flags, node builders, model builder helpers, parametrized and negative test patterns, and what NOT to do.
2. Write or modify the test file(s) following the patterns in that guide.
3. Build and run the tests to ensure all tests pass.

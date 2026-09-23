# Transformations Review

Apply to graph passes, pattern matching, pass infrastructure, and their tests
in common code, frontends, and plugins. Use the owning component's guidance
alongside this reference.

## Graph rewrites

- Check that matching predicates and replacement preconditions cover the
  intended operation versions, types, shapes, and attributes. Consider dynamic
  inputs and multiple consumers when the rewrite relies on static values or
  exclusive ownership of a subgraph.
- Verify numerical semantics, output types and shapes, and required runtime
  metadata and tensor/friendly names after replacement. Preserve control
  dependencies and stateful behavior when the affected graph uses them.
- Connect and rewire graph edges using `ov::Output<ov::Node>` values so the
  producer's output index is preserved. Passing a node as an input implicitly
  selects its default output (normally output 0), which can miswire a
  multi-output producer or fail if it has no default output. Preserve the
  original output or explicitly select `node->output(i)`. Node access through
  `get_node_shared_ptr()` remains appropriate for matching, inspecting
  attributes, copying metadata, and other operations on the node itself.
- Check pass registration and ordering so the intended graph form reaches the
  matcher. Preserve applicable pass-disable and plugin callback behavior, and
  report whether the graph changed according to the pass API contract.
- Keep frontend transformations independent of hardware and plugins.
  Put device-specific rewrites in the relevant plugin pipeline.

## Regression coverage

- Verify that a positive test demonstrates the intended rewrite. Add negative
  coverage when changed preconditions must reject an unsafe match, such as an
  incompatible operation version or an additional consumer.
- Check that comparisons cover the property being changed. In
  `TransformationTestsF`, `ATTRIBUTES` and `CONST_VALUES` are not enabled by
  default; enable them when the regression involves those values.
- Do not disable metadata or name checks merely to make a test pass. Establish
  that any changed metadata or naming behavior is intentional.
- Request numerical comparison when structural checks cannot establish the
  required equivalence; account for existing plugin accuracy coverage.

Consult the [transformation testing guide](../../../../src/common/transformations/docs/writing_tests.md)
for fixture behavior, comparator flags, and test placement.

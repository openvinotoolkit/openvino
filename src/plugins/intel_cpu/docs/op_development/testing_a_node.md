# Testing a New CPU Op

CPU functional tests are written the same way for any op — fixtures, parameter
sets, dynamic/static `InputShape`, the edge-case matrix, and ISA-selection checks
are all described in the [CPU functional tests guide](../../tests/README.md),
which is the source of truth. This page only lists what is specific to a *new* op.

For a new op, add two test files:

| File | Purpose |
|------|---------|
| `src/plugins/intel_cpu/tests/functional/shared_tests_instances/single_layer_tests/<op_name>.cpp` | Shared single-layer test instantiation (mandatory) — see [Shared single-layer tests](../../tests/README.md#shared-single-layer-tests). |
| `src/plugins/intel_cpu/tests/functional/custom/single_layer_tests/<op_name>.cpp` | Custom CPU-specific tests (recommended) — see [Custom CPU single-layer tests](../../tests/README.md#custom-cpu-single-layer-tests). |

Cover the [edge cases](../../tests/README.md#edge-cases-to-cover) relevant to the
op (precisions, layouts, static + dynamic shapes, rank and attribute variation),
and verify the right implementation is selected with `CheckPluginRelatedResults`
(see [Verifying implementation selection](../../tests/README.md#verifying-implementation-selection)).

If the op took the
[Eltwise fast path](./implementing_a_node.md#fast-path-routing-a-unary-elementwise-op-through-eltwise),
do **not** add a custom test file — extend the activation test infrastructure
instead, as described in
[Eltwise-routed ops](../../tests/README.md#eltwise-routed-ops-the-activation-test-infrastructure).

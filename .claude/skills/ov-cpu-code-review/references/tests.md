# CPU Plugin Tests Review

Applies to `src/plugins/intel_cpu/tests/` (`functional/custom/`,
`functional/shared_tests_instances/`, `unit/`) and to shared tests that CPU
instantiates. For new ops, also check
[add-cpu-op step 4](../../../../.github/agents-prototype/skills/add-cpu-op/step4-testing.md).
For transformation unit tests, apply the `ov-transformation-tests` skill and
`src/common/transformations/docs/writing_tests.md`.

## Reuse first

- **Mandatory.** Reuse rather than create. Before accepting a new test class,
  fixture, reference, or utility, check whether a shared single-layer or
  subgraph test class, `cpu_test_utils`, `filter_cpu_info`, an existing
  reference implementation, or an existing suite already covers it. Add
  instances to the existing suite instead of writing a new fixture. See "Check
  reuse before accepting new code" in [SKILL.md](../SKILL.md).

## Test structure

- **Mandatory.** When a CPU fusion produces an internal op from standard
  ops, test it through that subgraph, not by building the internal op
  directly. This exercises the real fusion, and the TEMPLATE plugin can
  compute the reference from standard ops. Assert that the fused node was
  created (`CheckNumberOfNodesWithType` / `ov::exec_model_info::LAYER_TYPE`).
  Build the internal op directly only when no fusion produces it, for example
  ops created by a frontend or GenAI (PagedAttention). Then the reference
  must be either an op that TEMPLATE can evaluate
  (`src/plugins/template/backend/ops/`) or an explicit `functionRefs` built
  from standard ops.
- **Mandatory.** Don't run the fusion manually in `SetUp`, and don't add
  `keep_const_precision` markup in the test to force a pattern. If the plugin
  doesn't fuse the pattern by itself, the pipeline has a defect (wrong pass
  placement, or earlier passes rewriting the pattern).
- **Mandatory.** Precommit CI runs CPU functional tests with
  `--gtest_filter=*smoke*`, so a functional suite runs in precommit only if
  its full name contains `smoke` (use the `smoke_` prefix). Suites without it
  (e.g. `nightly_`) run only in nightly or manually dispatched runs. If that
  is intentional, the PR must say why. Unit tests (`ov_cpu_unit_tests`) run
  unfiltered.
- **Mandatory.** New entries in
  `shared_tests_instances/skip_tests_config.cpp` need a ticket or TODO and a
  narrow regex. Check that no existing or new regex silently disables the new
  suite.
- **Mandatory.** Don't delete an existing regression test to get the PR
  green. Rename or fix it. If it failed, explain the failure.
- **Check.** Prefer kernel and executor coverage through functional or
  single-layer tests. Add kernel unit tests only for what a functional test
  can't reach. Don't add exhaustive conversion tests (e.g. all 65536 bf16
  encodings) for platforms where the CPU plugin never selects that
  precision.
- **Check.** Tests that CPU and GPU could share should go in the shared
  test classes.
- **Check.** Make the config a test parameter instead of branching inside
  the test body.
- **Check.** New test files use the concise nested form
  `namespace ov::test {`. Don't reformat existing files in unrelated PRs.
- **Check.** Place tests by architecture. Portable cases go in `common/`
  folders (`custom/single_layer_tests/instances/common`,
  `custom/subgraph_tests/src/common`) so that ARM CI runs them too. ISA-specific
  cases go in `x64/`, `arm/`, or `riscv64/`.

## Precision coverage

- **Mandatory.** Exercise bf16/f16 through `hint::inference_precision` in
  the test `configuration` (e.g. `cpu_bf16_plugin_config` from
  `cpu_test_utils.hpp`). Keep the model and IR precision at f32 or f16.
  `core_configuration` (`shared_tests_instances/core_config.cpp`) runs for
  every CPU functional test that derives from `SubgraphBaseTest`, `custom/`
  tests included. It forces `inference_precision=f32` unless the test sets
  the hint or uses `execution_mode=ACCURACY`. So a test that only builds a
  bf16 model doesn't cover the production bf16 path. Filter bf16/f16 configs
  by host ISA support (`ov::with_cpu_x86_bfloat16()`, `filterCPUInfo*`).
- **Check.** By default, `core_configuration` also converts bf16/f16 to f32
  in the reference model (`convert_precisions`). If a test disables that,
  uses `ACCURACY` mode, or provides its own reference, check that the
  reference supports mixed precision cases (e.g. f32 data with a bf16 gain).
  Otherwise the reference throws and the CPU result is never checked.
- **Check.** Loosened `abs_threshold` / `rel_threshold` values need a
  justification, should be scoped per precision (bf16/f16 vs f32), and must
  not hide a real accuracy bug.
- **Check.** For compressed weights, assert that the fused node's weight
  inputs keep the exact compressed type.
- **Check.** `GTEST_SKIP` must not hide a reachable failure, such as a
  configuration that `isSupportedOperation` accepts but `compile_model`
  rejects.
- **Check.** f32 data with a bf16 hint and similar hint/data mismatches are
  tested when the node picks a precision from the hint.

## Shapes and variants

- **Check.** Cover these shape cases:
  - uneven and odd sizes that don't align to SIMD width (e.g. a sequence
    length of 39, head sizes 3, 7, 17);
  - unequal head counts or dims where the op allows them (`K != V`,
    `qk_heads != v_heads`), or a negative test if it doesn't;
  - batch > 1;
  - lengths that stress the parallel split.
- **Check.** Dynamic-shape suites include a shape change between inferences
  that crosses tile and bucket boundaries, and a run after
  `release_memory()` where memory reuse matters.
- **Check.** Code paths that are gated only by host ISA (e.g. a fallback that
  runs only without AVX-512) need a test hook, or an explicit statement that
  they aren't covered on AVX-512 CI.
- **Check.** Add negative tests for invalid metadata (page tables,
  `subsequence_begins`, rejected group sizes) that the node validates.
- **Check.** Kernels with NaN/Inf-sensitive logic have NaN, Inf, and
  tiny-range tests.
- **Check.** Input data (`InputGenerateData` ranges, custom
  `generate_inputs`) matches the op's valid domain, reaches the edge regions,
  and is deterministic (no unseeded random sources), so failures reproduce.
- **Check.** Changes to node state, weights, or properties extend the
  export/import and model cache tests (`custom/behavior/`,
  `shared_tests_instances/behavior/`).
- **Check.** A suite that compiles one model per run shouldn't multiply runs
  over parameters that don't affect the outcome. Give a narrow check its own
  instantiation with only the relevant parameters.

## Test cost

- **Check.** Estimate the runtime of each new suite (instances × reference
  MACs). Drop redundant axes, such as modes that the fusion never emits,
  options that are always false, or configs that have no effect for a group
  size. Shrink shapes to the minimum that still hits each tile and bucket.
- **Check.** Machine-dependent cases (e.g. ones that hit a tile clamp only
  at 45–49 threads per stream) must pin the thread count through
  `ov::inference_num_threads`, or be removed. Comments must match what the
  test actually pins.

## Hygiene

- **Mandatory.** Remove debug artifacts from tests: `ov::serialize(...)` to
  the CWD, `std::cout`, and commented-out code. Remove unused locals, such as
  parameters created but never connected to the model, and unused includes.
- **Check.** Helpers use the parameters passed to them instead of re-reading
  globals or fixture fields.

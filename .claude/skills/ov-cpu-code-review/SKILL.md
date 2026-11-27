---
name: ov-cpu-code-review
description: Review OpenVINO Intel CPU plugin changes (src/plugins/intel_cpu) for node, executor, kernel (JIT/intrinsics/oneDNN), CPU transformation-pipeline, and test issues. Use when reviewing a CPU plugin pull request, branch, commit, or diff, or before marking your own CPU PR ready for review.
---

# OpenVINO CPU Plugin Code Review

## Purpose

Find CPU-plugin-specific defects that generic review misses: query-time
support contracts, precision negotiation against the host ISA, build-time vs
runtime ISA dispatch, shape inference and memory binding, scratchpad and
threading contracts, kernel bounds and performance, and test placement and
cost.

This skill extends [code-review](../code-review/SKILL.md). Load it first and
follow its review workflow, its evidence rules (step 5, "Challenge each
candidate finding"), its severity and confidence rules, and its
[delivery contract](../code-review/references/delivery.md). This file adds
the CPU checks and references. It does not replace those rules.

## Adding a new CPU op

When the PR adds a new CPU node or op, also check it against the
[add-cpu-op skill](../../../.github/agents-prototype/skills/add-cpu-op/SKILL.md):

- [step 1](../../../.github/agents-prototype/skills/add-cpu-op/step1-analysis.md):
  implementation strategy, precisions, layouts;
- [step 2](../../../.github/agents-prototype/skills/add-cpu-op/step2-implementation.md):
  node, registration, executor pattern;
- [step 3](../../../.github/agents-prototype/skills/add-cpu-op/step3-optimization.md):
  JIT/oneDNN path, `CpuParallel`;
- [step 4](../../../.github/agents-prototype/skills/add-cpu-op/step4-testing.md):
  tests.

Report deviations from it under the mandatory-rule policy of `code-review`.
For a new internal op, also review the core op (`src/core/`), its shape
inference, and its fusion transformation with the
[runtime](../code-review/references/runtime.md) and
[transformations](../code-review/references/transformations.md) references.

## CPU-specific workflow additions

Apply these during `code-review` steps 3 and 4.

### Map the CPU execution path

For each changed or new node, trace and record:

```
query_model / isSupportedOperation
  -> initSupportedPrimitiveDescriptors (port precisions, layouts)
  -> executor/impl selection (supports(), TypeMapping, priority)
  -> createPrimitive / prepareParams (shape-dependent setup, caches)
  -> execute / executeDynamicImpl (memory binding, scratchpad, parallel)
  -> kernel variant (build flags x runtime ISA x precision)
  -> fallback or explicit rejection
  -> tests that hit each variant
```

Build a matrix of **build configuration × host ISA × precision** for every
kernel:

- builds: default, `ENABLE_AVX512F=OFF`, ARM/RISC-V, `cross_compiled_file`
  variants;
- host ISA: SSE4.1, AVX2, AVX2_VNNI(_2), AVX-512, AVX-512 BF16/FP16, AMX;
- precision: f32, bf16, f16, plus compressed or quantized weights and
  KV cache.

Check that each reachable cell either runs a correct kernel or is rejected
before `compile_model` reports success. Don't infer one cell from a similar
one.

### Know the defaults that make paths reachable

- bf16/f16 run natively only when the host has native instructions for them.
  Otherwise the node runs in f32, and the data goes through reorders around
  it. Check `config.cpp` and `precision_support.cpp` at the reviewed revision.
- Functional test configs force f32 inference precision unless the test sets
  `inference_precision` or uses `ACCURACY` mode
  (`shared_tests_instances/core_config.cpp`). They also convert bf16/f16 to
  f32 in the reference model. A test that builds a bf16 model doesn't
  exercise the main bf16 path.
- Throughput mode multiplies per-stream resources (scratchpad, caches) by the
  number of streams.

### Check reuse before accepting new code

For every new piece of code in the PR, whether it's a helper, kernel,
executor, pass, pattern, or test class, search the repository for something
that already does the job. Do this for every component, not only kernels.
New code that duplicates an existing abstraction is a finding. The fix is to
reuse or extend the existing code, or, when a real gap exists, to share one
implementation between the new and old users.

Search at least these places:

| Component | Reuse before creating |
| --- | --- |
| Nodes | Similar nodes (FullyConnected, MatMul, GatherMatmul, ScaledDotProductAttention, PagedAttention, GatedDeltaNet) for lifecycle, memory binding, and state handling; `CPU_NODE_ASSERT` / `CPU_NODE_THROW`; `ShapeInferFactory` / `shape_inference.cpp` |
| Executors | `ExecutorFactory`, FC executors (`DnnlFCPrimitive`), `TypeMapping`, implementation lists, `VERIFY`, existing shared helpers (decompression params, row bucketing) |
| Kernels | `jit_load_emitter` / `jit_store_emitter`, existing emitters, `nodes/kernels/simd`, `scaled_attn/common.hpp` (`cvt_copy`, `dot_product`, horizontal sums), brgemm and attention helpers, `PlainTensor`, `CpuParallel`, the shared scratchpad, `utils/general_utils.h`, `ov::float16` / `ov::bfloat16` conversions |
| Transformations | Common passes (`convert_pagedattn_inputs`, `ConvertFullyConnectedToFullyConnectedCompressed`, `SymbolicOptimizations`), `CompressedWeightsBlock`, pattern predicates, `optional<>` / `operator\|`, existing CPU pipeline stages and graph-optimizer helpers |
| Tests | Shared single-layer and subgraph test classes (shared between CPU and GPU where possible), `cpu_test_utils`, `filter_cpu_info`, reference implementations, `TransformationTestsF`, existing suites that a new instance can extend |
| Build | Existing `sources.cmake` lists and `cross_compiled_file` entries |

When the PR adds a new abstraction anyway, the PR description should say
which existing component was considered and why it doesn't fit. If it
doesn't, ask.

### Evaluate performance and accuracy impact

For every PR, estimate how much the change can affect performance and
accuracy. Base the estimate on the code, not on the PR title.

The impact is likely big when the change does any of the following:

- adds or replaces a kernel, executor, or fusion on a hot path (attention,
  MatMul/FC, KV cache, MoE, recurrent or state-space ops), or changes
  implementation priority or selection;
- changes the default precision, precision mapping, quantization, or
  decompression, including KV cache or weight compression;
- changes accumulation order, the reduction method, approximations (exp,
  softmax, normalization), or the handling of NaN, Inf, and tails;
- changes threading, partitioning, tiling, scratchpad size, or memory reuse,
  or touches shared infrastructure that every node uses;
- adds work to every inference, such as validation loops, copies, reorders,
  locks, or shape-dependent setup moved into `execute()`;
- changes the transformation pipeline so that a fusion appears or disappears
  on real models;
- adds a new op or a newly enabled path for existing models.

When the impact is likely big, ask the author for results that match the
changed path, if the PR doesn't already include them:

- **Performance:** before/after numbers on the affected ISA. Use AVX2,
  AVX-512, or AMX as relevant, and ARM if the change touches it. Give
  `benchmark_app` latency and throughput for affected models. For LLMs, give
  first- and second-token latency for prefill and decode. For new kernels,
  add kernel-level timings compared with the previous or decomposed path.
  Include compile time and peak memory when setup or scratch changes, and
  multi-stream throughput when per-stream resources grow.
- **Accuracy:** before/after results on affected models, at each changed
  precision (f32 and the bf16/f16 hint). For LLMs, use a similarity or
  perplexity tool such as who_what_benchmark. Otherwise use the model's
  standard accuracy metric. State the threshold the results should meet.
  Functional tests with loosened thresholds are not enough.

Report missing results as **Needs verification**. Name the scenario, the
models or shapes, the ISA, and the metric that would resolve the concern.
Don't claim a regression or speedup that wasn't measured. When the estimated
impact is small, say so in one line and don't ask for measurements.

## Component references

Load the references that match the changed behavior:

| Changed paths or behavior | Reference |
| --- | --- |
| `src/plugins/intel_cpu/src/nodes/*.{h,cpp}`, `nodes_factory.cpp`, `cpu_types.*`, `shape_inference/`, executors (`nodes/executors/`), implementation lists and type mappings | [Nodes and executors](references/node.md) |
| `nodes/kernels/` (JIT, intrinsics, `scaled_attn`, `simd`, `x64`, `aarch64`), emitters, `cross_compiled_file` entries, `cpu_parallel.*`, `dnnl_scratch_pad.*`, memory blocks | [Kernels](references/kernels.md) |
| `src/plugins/intel_cpu/tests/` (functional `custom/`, `shared_tests_instances/`, `unit/`), shared tests used by CPU | [Tests](references/tests.md) |
| `src/plugins/intel_cpu/src/transformations/` (`transformation_pipeline.cpp`, `cpu_opset/`), `graph_optimizer.*`, common passes registered for CPU | [CPU pipeline and graph optimizer](references/pipeline.md), plus [transformations](../code-review/references/transformations.md) and the `ov-write-transformations` / `ov-transformation-tests` skills |
| `CMakeLists.txt`, `sources.cmake` under the CPU plugin | [CMake](../code-review/references/cmake.md), and the build-variant checks in [Kernels](references/kernels.md) |

The reference files contain two kinds of rules:

- **Mandatory** rules are CPU plugin conventions that reviewers enforce. A
  verified violation is at least a medium finding: cite the rule, state the
  consequence, and give the prescribed fix.
- **Check** items are investigative prompts. If you find no evidence for
  them, that doesn't make the code defective.

## CPU priority checklist (summary)

Use the references for detail. Check at least:

1. **Query-time contract.** Each op, mode, precision, and weight form that
   `isSupportedOperation` accepts has an implementation on every host it
   claims. Rejections happen at query time, not by throwing in
   `initSupportedPrimitiveDescriptors` or `compile_model`.
2. **Precision vs ISA.** bf16/f16 ports are selected only when the host can
   run them, using a `TypeMapping` with an f32 fallback. Compressed weights
   with f16 or non-native bf16 sources are handled.
3. **Build vs runtime ISA.** Kernel availability checks the compiled variant
   (`HAVE_AVX512F` etc.), not only `mayiuse()`. Stubs return a "not
   available" result instead of throwing, or computing nothing.
4. **Shape and bounds safety.** The kernels index raw memory without checks,
   so the node validates every input shape it relies on (ranks, leading dims,
   cache layouts, page tables) once per inference, before the kernel runs.
   Check arithmetic for overflow and narrowing (`int32` slot indices,
   `C*D` products).
5. **Shape inference and memory binding.** The node has a real
   `ShapeInferFactory`, not the clone-based fallback that marks ports as
   data-dependent. Edge memory is bound once. Shape-dependent setup goes in
   `prepareParams`.
6. **Scratchpad, threading, caches.** The node uses the shared scratchpad
   and `CpuParallel` correctly. Thread counts agree with the parallel
   partitioner. Buffers are sized from actual indexing bounds. Nested
   parallelism is avoided.
7. **Kernel quality and performance.** The kernel reuses `load`/`store`
   emitters, `nodes/kernels/simd`, and `scaled_attn/common.hpp` helpers. It
   has no hand-rolled conversions or tail loops, no latency-bound single
   accumulator chains, no huge compile-time unrolling, and no JIT where a
   simple memory-bound loop or Eltwise fusion is enough.
8. **Tests.** The tests follow [Tests](references/tests.md): `smoke_` prefix,
   subgraph-based tests for internal ops that check the fused node, the
   inference precision set via the config hint, small but targeted shape
   sets, no forced fusion in `SetUp`, and no deleted regression tests.
9. **Scope.** Unrelated refactors are split out (e.g. a standalone node's
   precision change, `scaled_attn` edits, or changes to a generic node made
   "for" a new kernel). So are changes to shared infrastructure that affect
   all nodes, unless they are justified and tested.
10. **Reuse rather than create.** New helpers, kernels, executors, passes,
    and test classes don't duplicate existing ones. See "Check reuse before
    accepting new code" above.
11. **Performance and accuracy impact.** You estimated the impact as described
    in "Evaluate performance and accuracy impact" above.
    If it is big, the PR includes before/after performance and accuracy
    results on the affected ISA and precisions, or the review asks for them.
12. **Hygiene.** No debug prints, `ov::serialize` calls, commented-out code,
    stale or overlong comments, or benchmark anecdotes left in comments.

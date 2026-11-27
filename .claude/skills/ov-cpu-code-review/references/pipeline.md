# CPU Pipeline and Graph Optimizer Review

Applies to `src/plugins/intel_cpu/src/transformations/`
(`transformation_pipeline.cpp`, `cpu_opset/`), `graph_optimizer.*`, and
common passes registered for CPU. Always also apply the
[transformations reference](../../code-review/references/transformations.md),
`src/common/transformations/docs/writing_transformations.md`, and the
`ov-write-transformations` / `ov-transformation-tests` skills. Violations of
those guides are mandatory findings.

## Pass placement

- **Mandatory.** Register a fusion at the pipeline stage where its pattern
  still exists. Earlier passes rewrite patterns:
  - `CommonOptimizations` (PreLpt) folds constant multiplies after a MatMul,
    scalar or per-output-channel, into the weights (`MatMulMultiplyFusion`);
  - `MoveEltwiseUpThroughDataMov` (PostLpt; the PerChannel variant is
    disabled on CPU and the Scalar variant has a callback) moves eltwise ops
    above data-movement ops such as the decompression Reshape, so fusions in
    PostLpt or later see the moved form;
  - `ConvertPrecision` changes the types of unmarked constants.
    `KeepConstPrecision` marks decompression constants so that
    `ConvertPrecision` preserves their low-precision type;
  - Snippets tokenization (`MainSnippets`) captures eltwise chains, so
    eltwise-based fusions must run before it.

  Verify that the fusion fires in the real pipeline, not only when a test
  runs it by hand. Register related fusions next to their siblings, e.g. in
  PreLpt next to `GatedDeltaNetFusion`.
- **Check.** Each pass that runs symbolic propagation on the whole model
  adds compile time on every model. Register the matchers in the existing
  PostLpt `SymbolicOptimizations` run instead. Some existing CPU fusions
  build their own internal `SymbolicOptimizations` (SDPA, MLP, QKV, causal
  mask). Don't copy that pattern.
- **Check.** Common passes that CPU disables or restricts
  (`CPU_DISABLE_PASS_*`, pass callbacks such as `KeepConstAndDecompression`
  or `isSupportedOperation`-based callbacks on fusions) still behave
  correctly after the change. A new common fusion may need to be disabled on
  CPU or get a callback, and node support checks belong in that callback
  through the node's `isSupportedOperation`.
- **Check.** The pass is registered for every architecture and precision it
  needs, and only those: `CPU_REGISTER_PASS_*`, `CPU_DISABLE_PASS_*`,
  `CPU_ENABLE_PASS_*`, and `CPU_SET_CALLBACK_*` for `COMMON`, `X64`, `X86`,
  `ARM`, `ARM64`, `ARM32`, and `RISCV64` (`transformations/defs.hpp`), plus
  AMX- or bf16-gated blocks. Check each combination where it runs
  or is skipped.
- **Check.** Fusions that touch FakeQuantize or dequantization subgraphs
  work with LPT both enabled and disabled (`useLpt`), and with the x64 and ARM
  LPT restrictions.
- **Check.** A CPU-only transformation that a GPU or paged op also needs
  probably belongs in a common pass, such as `convert_pagedattn_inputs` for
  paged cache precision. Precision preservation belongs in the fusion, or in
  the paged-fusion `enable_keep_const_precision()` marking. A separate
  "preserve precision" pass is usually the wrong fix.
- **Check.** An unconditional fusion can remove a previous fallback, for
  example models compiled with an explicit TURBO setting failing. Check every
  config that used to reach the unfused path.

## Compressed weights

- **Mandatory.** Follow the established compressed-op flow (FC,
  GatherMatmul):
  1. fuse the structure on the decompressed weights;
  2. add the op to `benefit_from_decompression` so `KeepConstPrecision`
     marks the constants;
  3. convert to the compressed form in a common `Convert<Op>ToCompressed`
     pass (`src/common/transformations/.../op_conversions/`) registered in
     `ConvertToCPUSpecificOpset` with the node's support check as callback,
     reusing `CompressedWeightsBlock` and
     `ConvertFullyConnectedToFullyConnectedCompressed::process_compressed_weights`.

  The new op must fit the existing order inside `ConvertToCPUSpecificOpset`
  (compressed conversions, then constant folding, then
  `MoveReadValueInputsToSubgraph`). `KeepConstPrecision` marking uses the
  FC supported-types list, so check that the new op's types are covered.

  Don't copy GPU compressed-weight macros, and don't set
  `keep_const_precision` as a side effect in callbacks.
- **Mandatory.** Expose the supported compressed weight types through a
  static node method (e.g. `getSupportedCompressedWeightsTypes()`) and pass
  them into the common pass as a parameter. Don't repeat the type list in
  the fusion and the node.
- **Check.** `copy_runtime_info` doesn't carry `KeepConstPrecision`, because
  `is_copyable()` is false. Mark the integer constants explicitly.
- **Check.** Make sure the conversion handles, or explicitly declines, scalar
  zero points and scales (per-tensor or legacy symmetric u4 `zp = 8`), float
  zero points, integer scales behind a `Convert`, and per-input-channel
  scales. Constructing the op without checking the stored layout lets a
  validation exception escape the matcher callback and abort `compile_model`.

## Matcher correctness in CPU passes

- **Mandatory.** Matcher callbacks must decline, by returning false, any
  match that the target op's validation would reject. Put the rank, type, and
  static-dim conditions in the pattern with predicates (`shape_matches`,
  `rank_more_than`, type predicates), not in the callback. An exception from
  `validate_and_infer_types` inside a callback aborts `compile_model` on
  models that compiled before.
- **Check.** Calling `get_length()` on a dim bound through `shape_matches`
  can hit a dynamic dim after `SymbolicOptimizations`. Require static dims
  in the pattern, or use `m.get_symbols()`.
- **Check.** Concat and other axes are stored raw: `-2` is not normalized to
  `2`. Match both forms (PyTorch exports often use negative axes) and test
  both.
- **Check.** Use `operator|` or `optional<>` instead of hand-rolled `Or`
  chains, `ov::op::vX` instead of `opsetX`, and the pattern API instead of
  `gen_pattern` outside tests.
- **Check.** `MATCHER_SCOPE` is present, so the pass works with selective
  build and ITT.
- **Check.** Passes that change model inputs or rewire nodes outside the
  match must be `ModelPass`es.
- **Check.** Comments that explain a pattern constraint (e.g.
  `consumers_count(1)`) give the real reason. Often that reason is a
  downstream CPU graph-optimizer rule, such as `RemoveMemoryInputConvert`
  followed by `MatchSdpaKvCache` rejecting extra consumers.

## Graph optimizer

- **Check.** `CPU_GRAPH_OPTIMIZER_SCOPE` inside a template can't tell
  instances apart. Put the scope in non-template wrappers and split the
  template into match and rewrite helpers. Give file-local helpers internal
  linkage.
- **Check.** Fusion passes set the rt_info marks that later passes or the
  graph optimizer rely on, and the transformation test checks them in the
  reference model. For example, `DisableInitSubgraphFusing` on the KV
  `ReadValue` is read by `MoveReadValueInputsToSubgraph` at the end of
  `ConvertToCPUSpecificOpset`.
- **Check.** State and memory-input rewiring keeps every consumer that the
  node's state lookup accepts. Otherwise the node fails at inference with
  null input states.

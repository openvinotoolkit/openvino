# CPU Nodes and Executors Review

Applies to `src/plugins/intel_cpu/src/nodes/`, `nodes/executors/`,
`nodes_factory.cpp`, `cpu_types.*`, and `shape_inference/`.

## Query-time support contract

- **Mandatory.** `isSupportedOperation()` must reject every configuration
  that has no implementation on the current build and host: modes,
  precisions, weight forms (constant vs non-constant, compressed type), and
  ISA requirements. If a check runs only for one branch (e.g. only inside
  `if (compressed)`), the other branches pass `query_model()` and then fail
  in `compile_model()`, which breaks AUTO/HETERO fallback.
- **Mandatory.** Don't throw `OPENVINO_THROW_NOT_IMPLEMENTED` from
  `initSupportedPrimitiveDescriptors()` for inputs that can be detected at
  query time, such as non-constant weights or scales. Move the check into
  `isSupportedOperation()`.
- **Check.** For every accepted configuration, `Executor::supports()`
  accepts at least one implementation. Compare the node's precision choice
  with the `supports()` conditions of each executor.
- **Check.** If the node is registered for x86 only in `nodes_factory.cpp`,
  its sources are listed in `CPU_PLUGIN_X64_SRCS` in `sources.cmake`, which
  is appended to `CPU_PLUGIN_SRCS` only under `if(X86_64)`. They must not be
  in the initial platform-common `set(CPU_PLUGIN_SRCS ...)` list. Headers that are
  x64-only don't need `OPENVINO_ARCH_X86_64` guards (dead macros).
- **Check.** `isSupportedOperation()` checks `op` for null before
  `ov::is_type`, or relies on a documented caller guarantee.

## Precision negotiation

- **Mandatory.** Choose port precisions with the executor `TypeMapping`
  mechanism (`TypeMaskAlias`, `bypass()`, `use<N>()`, `just<T>()`), as in
  `fullyconnected_implementations.cpp`. Don't hard-code a uniform precision
  from one port. Map hosts that can't run bf16/f16 natively to f32 instead
  of failing with "no supported implementation".
- **Check.** Compressed integer weights with an f16 source, or with bf16 on
  a host without native bf16: oneDNN FC enables weight decompression only on
  AVX2+ hosts with f32/bf16 sources. Route these cases to an f32 source while
  keeping the compressed weights, as `dnnlFCTypeMapping` does for bf16.
  Check that the case doesn't fall into a catch-all mapping that converts
  the weights to f32 too.
- **Check.** Match the kernel's bf16/f16 paths to the precisions the plugin
  actually selects (`utils/precision_support.cpp`):
  - bf16 is the default only with native AVX512_BF16, but an explicit bf16
    hint is also honored on plain AVX-512 (`avx512_core`) and AVX2_VNNI_2;
  - f16 needs `avx512_core_fp16`, AVX2_VNNI_2, or ARM fp16;
  - plain AVX2 never runs bf16/f16.

  A bf16/f16 kernel must either handle `avx512_core` without native bf16 by
  converting to f32 through the emitters, or reject it through the
  `TypeMapping`. Code for combinations the plugin never selects is dead
  complexity. Inside the kernel, compute in f32 and use the load/store
  emitters for the conversion.
- **Check.** If the node keeps a precision in the graph optimizer or the
  pipeline (e.g. keeps f16), confirm that the node supports it.
- **Check.** A KV/state cache precision taken from `inferencePrecision` can
  disagree with the op's "all float inputs share one type" rule when the data
  is f32 and the hint is bf16. Take the type from the data input when it is
  static.
- **Mandatory.** Use `ov::with_cpu_x86_*()` (`system_conf.hpp`) for feature
  checks in nodes and tests where a matching helper exists, instead of the
  oneDNN `mayiuse` API. Don't change `cpuUtils` from PRIVATE to PUBLIC just
  to reach ISA helpers from tests.

## Shape inference

- **Mandatory.** If the output shapes depend only on the input shapes, use a
  proper `ShapeInferFactory`: register the op in `shape_inference.cpp`, or
  write a small custom factory like `QSAIndexerShapeInferFactory`. Don't use
  the clone-based fallback or `InternalDynShapeInferFactory`. The fallback
  clones the op and re-validates it for every new shape. It also marks all
  ports as data-dependent, which makes the node a sync point in dynamic
  graphs.
- **Check.** Prefer `getOutputShapeAtPort` and shape inference results over
  hard-coding output dims from a particular input index.

## Input validation at execution

- **Mandatory.** Kernels index raw memory with `PlainTensor` or pointers,
  without bounds checks. The node must validate, once per inference and
  before the kernel runs, every cross-input shape relation that the kernel
  relies on. This covers: batch and head dims across inputs and states, cache
  ranks and page/head sizes for plain and quantized caches, ranks before
  `dims[i]` is read, and page-table and `subsequence_begins` coverage
  (`token_begins[0] == 0`, last entry equals `num_tokens`). If the core op
  checks only `compatible()` at compile time, dynamic batch changes can still
  cause heap overflows.
- **Check.** Put the checks in one validation point. Don't add them per
  token or per page inside kernels. If the core op spec or `shape_infer`
  guarantees an invariant, don't duplicate it in the kernel. Reference
  kernels can be more defensive, but optimized paths should drop the
  redundant guards.
- **Check.** Arithmetic on attributes or indices can overflow or narrow:
  `int64` products such as `C*D` (use `ov::util::mul_overflow`), flattened
  slot indices narrowed to `int32`, and signed/unsigned page ids.

## Node lifecycle and memory binding

- **Mandatory.** Bind edge memory once in `createPrimitive()`, as
  FullyConnected, MatMul, and GatherMatmul do. Edge memory objects don't
  change after `Graph::Allocate()`: reshapes update the descriptor in place,
  and user tensors swap the memory block. Don't re-bind in `execute()` or
  `prepareParams()`.
- **Check.** Put shape-dependent setup in `prepareParams()`, not
  `execute()`. Don't override `needPrepareParams()` when the base class
  check for changed input shapes is enough.
- **Check.** Remove unused members, for example a buffer that is declared
  but actually allocated on the scratchpad, and unused headers.
- **Check.** Use the naming style of the surrounding Node code: camelCase
  methods such as `bindMemoryArguments`.

## Errors and diagnostics

- **Mandatory.** In node implementations, use `CPU_NODE_ASSERT` and
  `CPU_NODE_THROW` (`node.h`) instead of `OPENVINO_ASSERT` and
  `OPENVINO_THROW`. They add the node type and name. Check every node the PR
  adds, not just the first one.
- **Check.** In executor `supports()` checks, use the `VERIFY` macro
  (`nodes/executors/debug_messages.hpp`) so the rejection reason is logged.
- **Check.** Error messages match their condition and print the operands
  that were compared.

## Executors and reuse of established abstractions

- **Check.** Compose GEMM-like projections from the existing FC executors
  (`ExecutorFactory<FCAttrs>`, `DnnlFCPrimitive`) instead of calling oneDNN
  matmul directly. Direct calls lose dynamic quantization, compressed-weight
  packing, and decompression attributes. If the PR doesn't compose them,
  ask for benchmark evidence against the decomposed path.
- **Check.** Nested FC executors receive the config's FC attributes
  (`dynamicQuantizationGroupSize`, `modelType`). Default `FCAttrs` silently
  disable them.
- **Check.** Duplicated helpers (decompression group dims and masks, row
  bucketing) should be shared through a common internal header. For row
  bucketing, reuse `utils::normalizeRowsBucket` (`dnnl/dnnl_utils.hpp`).
- **Check.** Packed-weight layouts from `format_tag::any` must not depend
  on whichever M value arrives first. Pick a fixed reference M, or keep a
  fallback if pd creation picks a different layout.
- **Check.** `implType()` reflects the implementation that actually runs.
  Don't report `jit_avx512` when oneDNN runs AMX brgemm underneath.
- **Check.** Executor cache keys hold everything that changes the generated
  code. Prefer storing the target ISA in each key.
- **Check.** Don't create primitives or allocate placeholder buffers just
  to query a scratchpad size. Lay buffers out at fixed offsets on the shared
  scratchpad instead.
- **Check.** Buffers on the shared scratchpad follow the scratchpad rules
  in [kernels.md](kernels.md) ("Scratchpad and memory footprint"): wrap them
  as `Memory` objects, because raw pointers kept across `update()` go stale.
- **Check.** Explain executor-variant selection criteria (e.g. in-place vs
  separate state) in a comment.
- **Check.** Follow the existing executor pattern of comparable ops:
  `gdn_implementations.cpp` lists a JIT executor and a reference executor
  of type Common (`ref/gdn_ref_executor`). A
  "Common" implementation is portable code with reasonable performance, but
  it shouldn't be hand-unrolled scalar code.

## Non-x86 executors

- **Check.** Executors in `acl/`, `aarch64/`, `kleidiai/`, `riscv64/`, and
  `mlas/` have their own per-backend type mappings and priorities. The
  mappings are defined next to `dnnl*TypeMapping` in
  `nodes/executors/*_implementations.cpp` (e.g. `aclFCTypeMapping` and
  `kleidiaiFCTypeMapping` in `fullyconnected_implementations.cpp`). Check that a change to the x86 mapping or support conditions
  has a matching change, or an explicit rejection, there. Remember that ARM
  defaults to f16 on fp16-capable hosts.

## Weights, in-place memory, and dynamic shapes

- **Check.** Packed or repacked weights go through the shared weights cache
  (`context->getWeightsCache()`), which is shared across streams. Their key
  covers the layout, precision, and ISA. Don't duplicate constant buffers for
  each stream.
- **Check.** In-place port configs, `resolveInPlaceEdges` overrides, and
  `isInPlace()` are correct. They change memory allocation and sync-point
  logic.
- **Check.** Dynamic-shape handling covers:
  - zero-dim and empty tensors (`isExecutable()` skips them, so `execute()`
    must not assume `prepareParams()` ran);
  - `needShapeInfer` overrides;
  - `executeDynamicImpl` delegating to `execute`;
  - the first inference after a shape change.
- **Check.** The node declares or rejects fusion correctly (`canFuse`, fused
  post-ops). Check whether Snippets can tokenize the op, because an op that
  Snippets wraps never reaches the node.

## Model caching and op attributes

- **Check.** Attributes that affect execution must survive
  `clone_with_new_inputs` and `visit_attributes`, so that model cache export
  and import work.

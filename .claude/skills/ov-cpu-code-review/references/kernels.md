# CPU Kernels Review

Applies to `src/plugins/intel_cpu/src/nodes/kernels/` (JIT, intrinsics,
`scaled_attn`, `sparse_attn`, `linear_attn`, `simd`, `x64`, `aarch64`,
`acl`, `kai`, `riscv64`, and top-level kernel files), emitters
(`emitters/plugin/`, `emitters/snippets/`), `cross_compiled_file` entries,
`cpu_parallel.*`, `dnnl_scratch_pad.h`, and memory blocks.

## Build-time vs runtime ISA dispatch

- **Mandatory.** Base kernel availability on both the runtime CPU and the
  compiled variant. A `cross_compiled_file` gets only the variants in its
  own `ARCH` list that are enabled in the build. With `ENABLE_AVX512F=OFF`
  (set explicitly, or forced by an old compiler), no AVX512F variant is
  built. The dispatcher then runs the AVX2 (or ANY) variant on AVX-512/AMX
  hosts, where `mayiuse(avx512_core)` and `mayiuse(amx_*)` still return
  true. This can cause three failures:
  - a stub throws, so `createPrimitive` fails;
  - the work split is planned for AMX items whose bodies are compiled out,
    so output rows are never written (silent wrong results);
  - kernels are JIT-built and then thrown away, which wastes compile time.

  Fix: stubs return `nullptr` or "not available", and the caller falls back.
  Alternatively, let the cross-compiled side report whether the kernel exists
  (true only under `HAVE_AVX512F`).
- **Check.** `cross_compiled_file` lists every variant that has code. The
  `HAVE_*` defines are cumulative (an AVX512F unit also defines `HAVE_AVX2`),
  so the dead code is an AVX2-only branch, such as `#elif defined(HAVE_AVX2)`
  or `!defined(HAVE_AVX512F)`. Without an `AVX2` entry, that branch is never
  compiled or tested, and AVX2-only hosts run the ANY variant.
- **Mandatory.** Inline helpers in headers that cross-compiled sources
  include must sit in the `ov::Extensions::Cpu::XARCH` namespace, as
  `scaled_attn/common.hpp` and `simd/simd.hpp` do. Otherwise copies compiled
  with different ISA flags violate the ODR, the linker keeps an arbitrary
  one, and lower-ISA hosts can crash with an illegal instruction.
- **Check.** Every ISA a kernel emits is guarded. An AVX2 or FMA kernel
  accepted on an SSE-only host crashes with an illegal instruction. Require
  the ISA at query time, or provide a lower-ISA fallback.
- **Check.** Kernels without x86-specific code don't need x86 `#ifdef`
  guards.

## Memory safety

- **Mandatory.** Size stack buffers from runtime dims, or bound them by a
  checked maximum. For example, `float buf[128]` overflows when the head size
  is above 128.
- **Check.** Strides and indexing: use the right dim for each tensor
  (`H*V` vs `H*K` strides when K != V), contiguous vs strided state access,
  and source vs destination in scalar fallbacks (`a[i]` vs `a_dst[i]`).
- **Check.** When a kernel dispatches on one input's precision and
  `reinterpret_cast`s the others to the same `T`, assert that all of them
  have that precision. Templated helpers that take `TA`/`TB` pointers need
  `static_assert`s on the types they reinterpret.
- **Check.** Score bucketing and histograms must handle NaN, ±Inf, and tiny
  ranges. Clamp both bounds before an `int` cast (casting NaN to an integer
  is UB). `nth_element` and `sort` need a strict weak ordering, which NaN
  breaks.
- **Check.** Tail lanes can't inject NaN or Inf into accumulators. Common
  causes are unfilled tail lanes (load emitters leave them untouched unless
  `is_fill` is set) and `0 * inf = NaN`. Document, for each ISA, where tail
  lanes are neutralized: when they are produced, or by masking when they are
  consumed.
- **Check.** Before dereferencing scratch pointers, check for early-exit
  conditions such as `rows == 0`.
- **Check.** Non-template functions defined in headers are `inline`.
  Function templates don't need `inline` (except full explicit
  specializations). Avoid `static` on header templates: it gives internal
  linkage and a separate copy in each translation unit.

## Threading

- **Mandatory.** Use `CpuParallel` for parallel sections in new kernels.
- **Check.** Per-thread buffers must be indexed the same way the
  partitioner counts threads. With the AUTO partitioner, which is chosen on
  some hybrid CPU configs or set by the user, the `ithr` passed to
  `CpuParallel::parallel_for*` lambdas goes up to `nthr * get_multiplier()`. With STATIC, the
  multiplier is forced to 1. Size per-thread buffers with
  `CpuParallel::get_num_threads()`, not `parallel_get_max_threads()`.
- **Check.** Avoid nested parallel regions. TBB tolerates them, but OMP
  builds need explicit handling. Merge adjacent parallel loops to cut
  scheduling overhead.
- **Check.** Behavior changes to `CpuParallel` (partitioner or multiplier
  semantics) affect every node. Require an explicit justification and
  coverage, or a separate PR.

## Scratchpad and memory footprint

- **Check.** Scratch size matches the actual indexing bounds and includes
  every buffer the code allocates. Watch for regions sized for the whole page
  table when only a subset of sequences (e.g. the prefill sequences) uses
  them. The shared scratchpad only grows, so peak usage stays allocated for
  each stream. Cap the size and fall back above the cap.
- **Check.** Tile sizes and scratch budgets should come from cache topology,
  or document the measured trade-off. Remember that throughput mode
  multiplies them by the number of streams.
- **Check.** If one node needs several live scratch regions, use
  `DnnlScratchPad::child(idx)` with stable indices for each extra region.
  Views of one buffer are fine. Don't give each nested executor its own
  ad-hoc buffer. A later request may reallocate a block: `Memory` objects
  follow it through `DnnlMemoryBlock::notifyUpdate`, but raw pointers must
  be re-read in `execute()`.

## Reuse established building blocks

- **Mandatory.** For JIT load/store with precision conversion and tails, use
  the `jit_load_emitter` and `jit_store_emitter` wrappers. They handle full
  vectors and tails through the same code path. The element count is a
  JIT-time constant, so use one emitter instance per distinct count, and pass
  `is_fill=true` with a neutral `fill_value` when tail lanes feed
  reductions. Don't hand-roll bf16/f16 ↔ f32 bit
  manipulation, scalar tail loops, or `clear_inactive_lanes`-style tail
  handling. Merge "full vector" and "tail" branches that differ only in the
  element count.
- **Check.** Before writing JIT for a simple memory-bound loop, consider
  intrinsics through `nodes/kernels/simd` (`load`/`store` per ISA,
  `simd_loop` for tails; it covers AVX2, AVX-512, and scalar, and works only
  from `cross_compiled_file` sources), or a graph rewrite to `Multiply`/`Add` that Eltwise
  fusion handles. JIT is overkill there and adds cache keys and maintenance
  code.
- **Check.** In C++ and reference kernels, reuse the helpers in
  `scaled_attn/common.hpp` (`cvt_copy`, `cvt_add`, `dot_product`,
  `multiply_scalar`) and the `ov::float16`/`ov::bfloat16` implicit
  conversions. Use the `utils/general_utils.h` helpers (`any_of`, `none_of`, `all_of`) and
  `PlainTensor` for strides and type wrapping. Don't template-dispatch over
  index types just to convert them.
- **Check.** Move helpers that other micro-kernels could use (l2norm, dot
  products, horizontal sums) into shared headers. Check whether they already
  exist.
- **Check.** Split quantized and non-quantized kernel interfaces so that a
  caller can't call the unscaled entry point on a quantized object, or the
  reverse. Don't break substitutability in a base class.

## JIT code generation quality

- **Check.** Structure: write a common `load -> compute -> store` workflow,
  with ISA differences confined to building blocks (`if constexpr` on the
  ISA inside helpers). Don't scatter store
  timing across ISA × mode branches.
- **Check.** Register contracts: keep the emitter-reserved vmm range in one
  named constant. Name scratch registers explicitly instead of borrowing
  `state_vmm`/`accumulator_vmm` under misleading names. Pass explicit
  indices instead of recovering them by dividing offsets.
- **Check.** Throughput: a single accumulator chain of dependent FMAs is
  latency-bound. Alternate two or more accumulators and fold them before the
  reduction. Prefer shuffle-based horizontal reductions to `vhaddps`.
  Hoist constant loads out of loops, into a GPR or a vector register set up
  once.
- **Check.** Unrolling: compile-time unrolling over the state or head
  dimension, with large caps (e.g. 4096), creates huge kernels and long JIT
  times. Use runtime loops above a threshold, or justify the cap in a
  comment. Consider moving outer loops (timesteps) into the kernel to avoid
  per-call overhead.
- **Check.** Pointer rewinds and loop invariants that depend on another
  function's advances should be computed or documented locally. Magic
  immediates (e.g. `0x55`) need a comment.
- **Check.** Hoist work that doesn't depend on the parallel index out of the
  parallel body. For example, per-timestep q/k normalization shouldn't be
  recomputed for every V lane.

## JIT kernel caching and emitter contracts

- **Check.** JIT kernels and executors built in `prepareParams` go through
  the runtime params cache (`src/plugins/intel_cpu/src/cache/multi_cache.h`). The key's hash and
  equality cover every JIT parameter: ISA, precisions, shapes that change the
  generated code, and tail counts.
- **Check.** Emitters declare correct `aux_vecs_count()` /
  `aux_gprs_count()` and respect the preserved-register lists. JIT kernels
  save callee-saved registers and issue `vzeroupper` where needed
  (`preamble()` / `postamble()`).
- **Check.** Intrinsic kernels that rely on flush-to-zero or
  denormals-are-zero behavior state it. The plugin's denormals optimization
  is a configurable property, not a guarantee.

## ARM and other architectures

- **Check.** Kernels changed for x64 have equivalent handling, or explicit
  rejection, on ARM: `kernels/aarch64`, ACL (`kernels/acl`), KleidiAI
  (`kernels/kai`), `emitters/plugin/aarch64`, and the `SVE` / `NEON_FP16`
  cross-compiled variants (dispatched by `with_cpu_sve()` /
  `with_cpu_neon_fp16()`). Check fp16 accumulation precision on ARM.
- **Check.** RISC-V kernels and emitters (`kernels/riscv64`,
  `emitters/plugin/riscv64`) and Snippets emitters (`emitters/snippets/`)
  follow the same ISA gating and tail-handling rules.

## Quantization and attention kernels

- **Check.** Quantization parameter layout and size assumptions are
  documented (`const float params[2]`). Derived sizes can't go negative,
  e.g. `dims[2] - 2*sizeof(float)`. Quantization is skipped where it doesn't
  pay off (e.g. `q_len == 1`).
- **Check.** Boolean conditions on precisions are correct: `!= u8 || != u4`
  is always true. Unused template parameters used only for SFINAE should
  instead bind the data types.
- **Check.** Copy-paste in unrolled group sums: each lane uses its own
  parameters (`b3[1]`, not `b0[1]`).
- **Check.** Property and feature names describe what is implemented. Avoid
  naming a feature after an algorithm it only partially implements.

## Performance claims

- Apply the `code-review` performance rules. For new kernels and executors,
  ask for kernel-level and end-to-end numbers on the target ISA, compared
  with the decomposed or reference path. Don't accept benchmark anecdotes in
  code comments in place of a design rationale.
- **Check.** Changes to generic nodes made "for" a new kernel (e.g. adding
  locks to every `Reorder::execute`) need a demonstrated root cause.
  Usually the fix belongs at the source.

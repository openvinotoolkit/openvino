# JIT kernel IR mode — implementation journal

Status snapshot as of 2026-04-17.

## What's implemented

### IR infrastructure (`jit_kernel_ir.hpp/cpp`)

- **Op tree with nested regions.** `Op` has an optional `body` (unique_ptr<IR>)
  for loops and branches. `region()` and `loop()` record nested bodies with
  cursor save/restore.
- **Sub-interval live ranges (LLVM naming).** `Interval` replaced with
  `Segment` + `LiveRange`. `compute_live_ranges()` uses per-recursion-level
  local maps — sibling branch bodies produce separate segments. `addSegment()`
  merges overlapping (not adjacent) segments. `liveAt()` provides O(log n)
  point queries via binary search.
- **LLVM-style interference-based allocator.** Each physical register tracks
  a segment union. Two values share a register iff their segments never
  overlap. Branch-local values (e.g., store_interleaved3 intermediates)
  naturally share registers across branches.
- **Integrated rematerialization.** When the allocator can't find a register,
  it rematerializes the longest-range interfering victim by cloning it at
  every use site. Supports both zero-input (broadcasts, constants) and
  **input-aware** remat (ops whose inputs all outlive the victim). Clones
  carry the victim's `reads` for correct lowering.
- **Debug dump** shows op names, nesting, live ranges (with segments), and
  register assignments via `OV_JIT_IR_DUMP=1`.

### DSL integration (`jit_kernel.hpp`)

- `begin_ir()` / `end_ir()` bracket IR-mode recording. Controlled by
  `OV_JIT_IR_MODE=1` env var.
- `ir_def<N>()`, `ir_use()`, `ir_load<N>()`, `ir_store<N>()`,
  `ir_broadcast<N>()`, `ir_cmp()`, `ir_if()` bridge DSL calls to IR ops.
- `vec_op()`, `vec_copy()`, `vec_permute()` dispatch between IR/eager modes.
- `foreach` in IR mode: loop counter is a GPR (not IR-managed), loop header/
  footer are emit closures, body records into a nested `loop()` region.
- `variable::operator=(variable&&)` transfers both `_reg` and `_vid`.
- `_vid` is mutable (same rationale as `_reg`).

### Predicated loops (`foreach_predicated`)

- **Single-loop tail handling.** `foreach_predicated<N>` computes a
  k-register mask per iteration from the remaining element count. Full
  iterations use all-ones mask (zero overhead). The last iteration uses a
  partial mask. No separate tail code path.
- **Ambient predication.** `_predicated` flag + `_active_mask` on the kernel.
  When set by `foreach_predicated`, `ir_load` and `ir_store` automatically
  emit masked instructions. Wrappers like `store_interleaved3` need no
  `_masked` variants — they call `ir_store` which picks up the mask.
- **Mask setup emitted at lowering time** via `ir_use` closure (not at
  recording time). Correctly runs inside each loop iteration.

### Type-converting load/store

- **`ir_load` / `ir_store` dispatch on pointer element type** via
  `if constexpr`. One function, multiple behaviors:
  - `float*` → `vmovups` (direct)
  - `uint8_t*` load → `vpmovzxbd` + `vcvtdq2ps` (zero-extend + int→float)
  - `uint8_t*` store → `vcvtps2dq` + `vpmovusdb` (float→int + saturate+pack)
- **Masked variants** follow the same dispatch. AVX-512 `vpmovzxbd` and
  `vpmovusdb` natively support k-register masking.
- **No `jit_load_emitter` / `jit_store_emitter` needed** for these types.
  The DSL emits 2 instructions directly — simpler and faster than the
  general-purpose emitter infrastructure.

### color_convert (`color_convert.cpp`)

- **Unified f32 and u8 path.** The `if constexpr` split between f32 and u8
  is eliminated. One loop body handles both types — `T` flows through
  `ir_load`/`ir_store` dispatch. The BT.601 math (broadcasts, subtract,
  multiply, FMA, clamp) is pure `float[N]` regardless of `T`.
- **No tail code.** `foreach_predicated` handles all pixels including the
  remainder. The eager-mode tail (`_if(width != 0)._then(...)`) is removed
  for f32. For u8, the entire eager path (main loop + tail) is replaced by
  a single `foreach_predicated` with type-converting load/store.
- Full BT.601 NV12→RGB/BGR: 8 broadcast coefficients + 1 xor zero, FMA
  chain, clamp, conditional store via `ir_if`.

## Bugs found and fixed (10 total)

See `analysis_nv12_f32.md` for detailed analysis of bugs #1–9.

1. Loop interval extension too aggressive — extended intra-iteration values.
2. Cursor clobbered by nested regions (ir_if inside foreach).
3. Move assignment didn't transfer `_vid`.
4. `uni_vpermps` allocates scratch inside emit closure.
5. Broadcast address captures dead GPR.
6. Remat evicts loop-carried values without forward use (wrap-around).
7. Remat clone interval too long (used original end instead of actual last read).
8. Clone interval off-by-one after remat op insertion.
9. Dangling reference after `intervals.push_back()`.
10. **vpermps used `rax` as scratch GPR** — `rax` is in the GPR allocable pool,
    so it can be the loop counter or any other variable. Fixed by using `param1`
    (rdi on Linux), which is excluded from the pool.

## RoPE kernel IR port

### New files
- `rope_kernel_ir.hpp` / `rope_kernel_ir.cpp` — IR-mode RoPE kernel
  using `jit_kernel` DSL. Drop-in replacement for `jit_rotary_kernel`.
- Selected at runtime via `OV_JIT_IR_ROPE=1` env var in `rope.cpp`.

### Generated code comparison (Llama2, rotary_ndims=128, f32, AVX-512)

| | Legacy | IR (no unroll) | IR (unroll=4) | Clang -O3 |
|---|--------|---------------|--------------|-----------|
| Code size | 1088 bytes | 227 bytes | 711 bytes | ~280 bytes |
| Data tables | 640 bytes | 0 | 0 | 0 |
| Vector insns/iter | 12 | 12 | 12 | 10* |
| Registers | 5 (zmm0-4) | 4 (zmm0-3) | 4 | 3 |
| Tail handling | none | predicated | predicated | predicated |
| FMA copies | 0 | 0 | 0 | 0 |

\* Clang folds loads into FMA memory operands (2 fewer instructions).

### New IR primitives added
- `ir_load<N>(ptr, byte_offset=0)` — type-converting (f32/u8/f16/bf16),
  masked/unmasked via ambient `_predicated` flag, zero-masking `{k1}{z}`.
- `ir_store(ptr, byte_offset, val)` — same type dispatch and masking.
- `Insn3::fmsub231ps` + `fmsub(a, b, c)` DSL wrapper.
- `deinterleave2(a, b)` → `(evens, odds)` — portable semantic op.
  x86: vperm2i128+vshufps (AVX2) or vshuff32x4+vshufps (AVX-512).
  ARM would lower to UZP1/UZP2.
- `interleave2(evens, odds)` → `(lo, hi)` — inverse.
  ARM would lower to ZIP1/ZIP2.
- `begin_ir(force=true)` — bypasses `OV_JIT_IR_MODE` env var check.
- `foreach_predicated<N>(count, body, unroll=1)` — manual unroll factor.

### Allocator improvements
- **Half-open intervals** `[start, end)` — LLVM convention. Enables
  correct coalescing without special-case overlap logic.
- **LLVM early/late slot model** — each op N occupies slots 2N (reads)
  and 2N+1 (defs). Reads end before defs start at the same instruction.
- **Tied operands** — `Op::tied_to` field (LLVM-style operand constraint).
  `def_tied()` IR builder. Allocator coalesces via `copy_of` mechanism.
- **FMA coalescing** — `fma`/`fnma`/`fmsub` no longer emit `vec_copy`.
  The emit closure skips vmovups when `def == reads[0]` (coalesced).
- **vpermps split** — separated into two IR ops (table load + permute)
  so the allocator sees both inputs and assigns distinct registers.

### Bugs found and fixed (session)
11. **Pointer advance outside IR** — eager `src += ...` executed at
    recording time, not lowering time. All iterations loaded from the
    final pointer position. Fix: wrap in `ir_use` closures.
12. **Tail not handled** — `for (i < half/N)` skipped remainder elements
    (QwenVL: half=40, N=16, 8 elements lost). Fix: replaced manual loop
    with `foreach_predicated`.
13. **begin_ir() gated by env var** — IR RoPE kernel ran in eager mode
    when `OV_JIT_IR_MODE` not set, producing correct results by accident.
    Fix: `begin_ir(true)` forces IR mode.

## Current test results

### Passing
- All 29 `JitKernelIR.*` unit tests, including:
  - LLVM-style half-open intervals with early/late slots
  - Tied-operand coalescing for FMA
  - Sub-interval live range computation
  - Integrated remat (zero-input and input-aware)
  - End-to-end kernels: vec_add, vec_expr, fma, foreach, if/else,
    store_interleaved3, foreach+if/else+interleave3, foreach_predicated
- All 4 `JitKernel.*` unit tests.
- All 19 `smoke_RoPETest*` functional tests with `OV_JIT_IR_ROPE=1`.
- All 8 `smoke_TestsConvertColorI420*` functional tests with IR mode.

### Known limitations
- Pre-existing u8 accuracy failure (144×16 single test). Unrelated.
- IR-level loop unrolling pass (option B) implemented but loop counter
  adjustment incomplete. Manual unrolling via `foreach_predicated` unroll
  parameter works correctly.
- Memory-operand folding (load into FMA) not implemented. Clang achieves
  10 insns/iter vs our 12 by folding cos/sin loads into FMA operands.

## Open work

### Immediate
- Port `rotary_interleave_ir` to use `foreach_predicated` with unrolling
  (currently uses manual unrolled loop).
- Add bf16/f16 functional test coverage for RoPE IR kernel.

### Allocator
- Memory-operand folding — model instructions explicitly so lowering can
  fold single-use loads into consuming instructions.
- IR-level loop unrolling pass — needs loop counter adjustment
  infrastructure (iteration count as modifiable GPR).

### Next kernels
- **I420 converter**: same structure as NV12 but separate U/V planes.
- **RoPE interleaved**: already has `deinterleave2`/`interleave2` ops.

### Future
- Chained rematerialization.
- Spill/reload for non-rematerializable values.
- AVX2 `foreach_predicated` variant.
- TwoAddressInstructionPass equivalent (explicit COPY insertion before
  tied ops, removed by coalescer — full LLVM alignment).

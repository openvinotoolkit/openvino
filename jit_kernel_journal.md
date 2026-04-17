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

## Current test results

### Passing
- All 29 `JitKernelIR.*` unit tests, including:
  - Sub-interval live range computation (per-branch segments, addSegment
    merge policy, liveAt binary search)
  - LLVM-style interference-based allocation
  - Integrated remat (zero-input and input-aware)
  - `EndToEndForeachPredicated` — predicated loop with tail (25 elements,
    N=16: 1 full + 1 partial iteration, no overwrite past count)
  - End-to-end kernels: vec_add, vec_expr, fma, foreach, if/else,
    store_interleaved3, foreach+if/else+interleave3
- All 4 `JitKernel.*` unit tests.
- All 8 `smoke_TestsConvertColorI420*` functional tests — both f32 and u8
  now use IR mode + `foreach_predicated`. No tail. No eager fallback.

### Known limitations
- The `EndToEndForeachIfElseStoreInterleaved3` unit test kernel still
  overflows the register pool: 9 constants + 3 clamped values exceed 16
  registers even after integrated remat.
- Multi-iteration foreach+branches has a pointer-advance bug (second
  iteration outputs zeros). Pre-existing, not caused by allocator changes.
- Pre-existing u8 accuracy crash (`munmap_chunk(): invalid pointer` on
  144×16), reproduces without IR mode. Unrelated.

## Open work

### Immediate
- Diagnose the foreach+branches multi-iteration pointer-advance bug.
- Investigate whether the color_convert kernel fits with `OV_JIT_IR_MODE=1`
  for width=32 (2 iterations). Currently width=10 (1 iteration) passes.

### Hardening
- Unit tests for: loop interval extension edge cases, cursor save/restore,
  move-assignment `_vid`.
- Replace `push(param1)/pop(param1)` in vpermps with a cleaner solution
  (GPR IR pool, RIP-relative, or pre-loaded loop-invariant tables).

### Next kernels
- **RoPE kernel** (`rope_kernel.cpp`): needs bf16/f16 type-converting
  load/store (same `if constexpr` pattern as u8), `ir_load` with byte
  offset, and custom deinterleave/re-interleave shuffle DSL ops.
- **I420 converter**: same structure as NV12 but separate U/V planes.

### Future
- Chained rematerialization (remat an op by first rematerializing its
  inputs recursively).
- Spill/reload for non-rematerializable values under extreme pressure.
- AVX2 `foreach_predicated` variant: main loop (unmasked) + masked tail,
  decided at recording time (same IR body, different loop structure).

# JIT kernel IR mode — implementation journal

Status snapshot as of 2026-04-12.

## What's implemented

### IR infrastructure (`jit_kernel_ir.hpp/cpp`)

- **Op tree with nested regions.** `Op` has an optional `body` (unique_ptr<IR>)
  for loops and branches. `region()` and `loop()` record nested bodies with
  cursor save/restore.
- **Linear-scan allocator** with trivial coalescing (copy ops) and
  rematerialization (evict + clone cheap ops like broadcasts).
- **Interval computation** walks the op tree recursively. Loop bodies extend
  intervals of values defined before the loop and used inside it. Branch
  bodies (non-loop regions) use sequential indexing — no extension.
- **Rematerialization** handles wrap-around search (retry from index 0 for
  loop-carried values), clone interval based on actual last read (not
  original end), and recursive rewrite through nested REGION bodies.
- **Debug dump** shows op names, nesting, intervals, and register assignments
  via `OV_JIT_IR_DUMP=1`.

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

### color_convert f32 path (`color_convert.cpp`)

- Full BT.601 NV12→RGB/BGR conversion in IR mode: 8 broadcast coefficients +
  1 xor zero, FMA chain, clamp, conditional store via `ir_if`.
- Tail handling: eager-mode masked load/store for `width % N` remaining pixels.
- Broadcasts use `_consts.reg()` directly (raw RegExp) to avoid dangling GPR
  addresses from temporary variables.

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
- All 13 `JitKernelIR.*` unit tests.
- All width=10 f32 smoke tests (4/4) — but these exercise only the eager tail
  on AVX-512 (N=16, `10 >> 4 = 0` loop iterations).
- 1Plain/RGB width=32 (1/1) — exercises the IR loop (2 iterations).
- All width=10 u8 smoke tests (4/4).

### Failing
- 3 of 4 width=32 f32 tests fail: 1Plain/BGR, 3Plains/RGB, 3Plains/BGR.
  - 1Plain/RGB passes → the then-branch (store r,g,b order) works.
  - BGR failures → the else-branch or the second loop iteration has a bug.
  - Error values are large but clamped (e.g. "Expected: 0 Actual: 255"),
    suggesting register assignment confusion, not uninitialized memory.
- Pre-existing u8 accuracy crash (`munmap_chunk(): invalid pointer` on 144×16),
  reproduces without IR mode. Unrelated.

### Root cause of remaining width=32 failures

Not yet diagnosed. The `param1` fix resolved the `rax` clobbering but didn't
fix all cases. Likely candidates:

1. **Register conflict in the else-branch.** The allocator treats then/else as
   sequential code. Values computed before the if/else must survive into both
   branches. If the allocator frees a register too early (because the then-branch
   "consumed" its last indexed use), the else-branch reads stale data.
2. **Remat interaction with branches.** A rematerialized coefficient's clone
   may land inside one branch but not the other, leaving the original (with
   truncated interval and freed register) as the source in the other branch.
3. **vpermps constant table** loading — the `push(param1)/pop(param1)` pattern
   is correct in isolation but may interact poorly with nested regions.

## Open work

### Immediate (blocking correctness)
- Diagnose and fix the 3 remaining width=32 f32 failures.

### Hardening
- Unit tests for: loop interval extension, cursor save/restore, move-assignment
  `_vid`, remat wrap-around, recursive rewrite through nested bodies.
- Replace `push(param1)/pop(param1)` in vpermps with a cleaner solution
  (GPR IR pool, RIP-relative, or pre-loaded loop-invariant tables).

### Future
- u8 path port to IR mode.
- I420 converter.
- Tail handling inside IR (masked IR loads/stores) instead of eager fallback.
- Pre-existing u8 accuracy crash investigation.

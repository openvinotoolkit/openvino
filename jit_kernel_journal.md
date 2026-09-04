# JIT kernel IR mode — implementation journal

Status snapshot as of 2026-09-03, after the review pass described at the
bottom. Describes what the code on this branch actually does.

Sources of truth:
- `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel_ir.{hpp,cpp}` — IR, passes, allocator, verifier (475 + 1025 lines)
- `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel.{hpp,cpp}` — DSL + lowering closures (2662 + 702 lines)
- `src/plugins/intel_cpu/src/nodes/kernels/x64/rope_kernel_ir.{hpp,cpp}` — RoPE kernel on IR mode
- `src/plugins/intel_cpu/src/nodes/color_convert.cpp` — NV12/I420 converters on IR mode
- `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel_target.{hpp,cpp}` — target capability queries (TTI analogue)
- `src/plugins/intel_cpu/tests/unit/jit_kernel_ir_test.cpp` — 43 tests
- `src/plugins/intel_cpu/tests/unit/jit_kernel_test.cpp` — 4 tests

## Where the implementation stands relative to the design docs

Constraints from `jit_kernel.md` that no longer hold:

| Doc statement | Reality |
|---|---|
| "One DSL call = one emitted instruction" | `ir_load`/`ir_store` emit 1–3 (type conversion), `store_interleaved3` ~9, partial access expands into a stack bounce plus a copy loop. The surviving rule: the author picks the operation, the DSL never re-decides it based on liveness. |
| "Eager mode stays the default, IR mode is opt-in per kernel" | Eager *vector* mode is gone. Vector primitives require an active IR. Eager scalar/GPR code — `var<size_t>`, `foreach`, `_if/_then/_else`, `stack_frame`, legacy emitter interop — remains and is used by `cpu_convert.cpp`. |
| "IR is a register allocator, not an optimizer" | True again: the pipeline is two-address lowering, liveness, allocation with rematerialization, verification, lowering. The loop-unrolling pass was removed (see below). |
| "No auto-spill" | Still true, and now enforced loudly: exhaustion throws. |

## Architecture as implemented

### Pass pipeline (`build_default_pipeline`)

```
TwoAddressPass      → COPY insertion before tied-operand ops (breaks SSA, LLVM-style)
LiveRangeAnalysis   → CFG + dataflow liveness → live ranges
DumpPass("before")  → OV_JIT_IR_DUMP
RegisterAllocator   → interference assignment; may rematerialize, then re-runs analysis
DumpPass("after")
VerifyPass          → asserts the invariants lowering depends on
LoweringPass        → ctx.lower_fn(ir, assignment)
```

`PassContext` carries the three allocation orders, analysis results, the
assignment, the lowering callback and the debug flags.

### IR shape

`Op`: `reads`, `def`, `emit` closure, `is_copy`, `tied_to`, `def_rc`
(register class), `early_clobber`, optional nested `body` (region),
`is_loop`, debug `name`.
`IR` holds a `std::list<Op>` plus a cursor used by `region()`/`loop()` to
record nested bodies. Control flow is structural nesting (MLIR-shaped).

Value ids are dense and SSA at record time; `TwoAddressPass` deliberately
breaks that afterwards (one id, several defs), matching LLVM's
post-two-address form, and `LiveRange::valnos` carries the value numbers.

### Liveness — CFG plus dataflow

`compute_live_ranges` builds a real CFG from the region tree and solves

```
live_out[B] = U live_in[S], S in succ(B)
live_in[B]  = uses[B] U (live_out[B] - defs[B])
```

to a fixpoint, then converts block-level liveness into segments.

- Loop regions get their own header block: the back edge targets the
  compare, not the code that initialised the counter. Without that split,
  a value defined before a loop and used after it would look dead inside
  the body.
- Conditional regions have edges header→body and header→join, so a value
  needed after the region stays live along both paths.
- Every op owns one index, region ops included — a region op is the branch
  instruction of its header block, so its reads (loop counter, bound) are
  live exactly where the compare reads them.
- Blocks are created in walk order, so consecutive blocks are adjacent in
  slot space and `addSegment` merges pass-through liveness into single
  segments.
- Values that die inside a branch body keep disjoint ranges, so
  branch-local temporaries still share registers.

Slot model: op `N` occupies early slot `2N` (reads) and late slot `2N+1`
(defs), so a read and a def at the same op do not overlap and tied
operands/copies coalesce.

### Register classes

Three: `Vec`, `GPR`, `Mask`. The class lives on the virtual register, and
each has an allocation order — the physical registers it may use, in
preference order (LLVM's `AllocationOrder`). Predicates are a normal class,
which is what removed the last hardcoded register from the DSL: masks used
to be `Opmask(1)` everywhere, so two mask-consuming constructs in one
kernel would have collided silently.

Per-target predicate constraints live in the order, not in the allocator:
x86 lists `k1..k7` because `k0` cannot encode a write-mask (LLVM spells
this `VK*WM`), SVE would list `p0..p7` for governing predicates (`PPR_3b`),
and RVV would list only `v0` (`VMV0`). ISAs with no predicate file return
an empty order, so requesting a mask fails loudly.

### Early-clobber defs

`Op::early_clobber` marks a def that is written before the op's reads are
consumed — a multi-instruction expansion that scribbles on its
destination first. Such a def takes the *early* slot instead of the late
one, so it interferes with its own reads and the allocator cannot give it
a read's register. Same concept and purpose as LLVM's early-clobber
operand.

This is not theoretical: the active-lane-mask computation
(`xor bits; bts bits, count; dec bits`) got the count's register the first
time it was written, because the count dies at that op and reads/defs
normally do not interfere. The differential test caught it as wrong
output.

### Register assignment (`assign_registers`)

Not linear scan and not named as if it were: live ranges are visited in
order of first definition, and each takes the first register in its class's
allocation order whose already-assigned segments do not overlap
(the interference test LLVM's allocators use, with first-fit selection).

- Three allocation orders (`vec_pool_indices`, `gpr_pool_indices`,
  `mask_pool_indices`), explicit lists. Registers reserved eagerly
  (`arg()`, `reserve<>()`, the constants-table register) are absent from
  them, so the allocator never hands out a register the kernel holds.
- Coalescing: a value with `copy_of` set takes its source's register when
  that does not interfere; lowering then skips the copy entirely.
- On failure: rematerialize (def with no reads, or one whose reads all
  outlive the victim), which rewrites the IR and returns `std::nullopt` so
  the caller re-runs analysis. If nothing is rematerializable, throw
  `allocation_failure` naming the class, pool size, value, slot and how
  many interfering values were live.
- **No spiller.** Frame sizing happens before the pipeline runs and the IR
  layer has no hook to emit a store/load of a physical register, so a
  spiller needs both of those first. Faking it is what the previous stub
  did, and it emitted nothing at all.

### Vector register file and the EVEX constraint

The eager pool is the low 16 vector registers. `zmm16..zmm31` exist on
AVX-512 but are EVEX-only, so VEX/SSE-encoded instructions cannot name
them — handing one to eager code produced Xbyak's "not supported".

Kernels therefore declare their width: `set_vec_width(512)` (as
`color_convert` and `rope_kernel_ir` do, computed from `N`) extends the IR
allocation order to all 32 registers. Anything else keeps 16. This is the
VEX/EVEX register-class distinction LLVM models as separate register
classes, expressed here as one opt-in per kernel.

Measured effect on `color_convert` NV12 f32: 153 values allocated with 32
registers and no rematerialization, against 6 remat clones at 16.

### Verifier

`verify()` + `VerifyPass` run unconditionally between allocation and
lowering and throw `verification_failure` on:

1. malformed live ranges (empty, unsorted or overlapping segments, bad valno)
2. reads of values nothing defines (a botched IR rewrite)
3. values referenced by an op but left unassigned (lowering would throw
   `std::out_of_range` with no context)
4. two values sharing a physical register with overlapping live ranges —
   the property the allocator exists to guarantee
5. leftover tied operands after `TwoAddressPass`, copies with the wrong
   arity, registers outside their pool

Not checked: that every read is *dominated* by a def. A value defined only
inside a conditional region and used after it is invalid IR that the
verifier still accepts; dataflow over-approximates its range (live-in at
entry) rather than producing a hole, so the result is conservative rather
than wrong.

### Lowering

Tree walk. For each op, resolve `reads`/`def` to physical registers and
call the emit closure. Region ops emit their header closure and then
recurse. `is_copy` ops are emitted by the lowering walk itself
(class-dispatched `mov` / `uni_vmovups`) and skipped when coalesced.

Emit closures must not allocate: they run after allocation, so anything
they need arrives through `EmitContext`, is captured as an immediate, or
lives in a register outside both pools. Scratch registers are IR values —
no closure pushes to the stack any more.

### GPR management

GPRs are IR-managed. `ir_def_gpr` records a GPR value;
`ir_shr/ir_and/ir_add/ir_imul` are IRBuilder-style helpers; `ir_alloca`
returns a GPR value holding `lea rsp+offset`; `arg()` inside IR mode
records parameter loads as GPR defs; `foreach` records its counter as a GPR
value and its header/footer as region reads plus emit closures.

The pool is small (13 allocable on this host), so GPR-heavy expansions
(`ir_load_partial`, `ir_memcpy`) can exhaust it — with no spiller that is a
hard build failure, which is why `rotary_half_ir` still refuses shapes that
need an epilogue.

### Active length, and who decides how tails are handled

Memory operations take an explicit active length, `jit_kernel::vlen`,
shaped like LLVM's vector-predication operands — a mask and a length,
either of which may be absent, and neither of which names a register type:

- `vlen::all()` — every lane
- `vlen::elements(count)` — `count` leading lanes, count in a GPR value
- `vlen::predicated(mask, count)` — a materialized predicate value (an IR
  value in the `Mask` class), with the count kept alongside for operations
  that cannot consume a predicate
- plus a `terminal` bit: last iteration of the loop, so pointer bumps can
  be skipped

`foreach_vec<N>(count, body)` is the loop the kernel author writes. How
leftover elements are handled is the target's decision, mirroring LLVM's
`TailFoldingStyle`:

| Style | Shape | Targets |
|---|---|---|
| `mask` | one predicated loop | AVX-512, SVE |
| `epilogue` | full-width loop + counted tail | AVX2, SSE, NEON |
| `length` | one loop, `vl` set per iteration | RVV |

`length` is declared and unimplemented: `vl` is machine state, so it needs
a RISC-V generator plus `vsetvli` insertion in the loop header (LLVM models
this with implicit `VL`/`VTYPE` operands and the `RISCVInsertVSETVLI`
dataflow pass). `foreach_vec` throws rather than approximating it.

Memory operations then realize the active length once, in one place:
predicate it if the target can (`supports_masked_access`), otherwise
scalarize through a stack slot — the DSL's equivalent of LLVM's
`ScalarizeMaskedMemIntrin`. Predicates are memoized per body instance, so
several accesses sharing one active length share one `kmov`.

`OV_JIT_TAIL_FOLDING=epilogue|mask|length` overrides the choice — the
counterpart of LLVM's `-prefer-predicate-over-epilogue`, and the reason
the epilogue path stays tested on an AVX-512 host.

Measured on this host (AVX-512), same kernels, both strategies passing:

| Kernel | `mask` | `epilogue` | legacy |
|---|---|---|---|
| NV12→RGB f32 converter | 833 B | 1309 B | — |
| RoPE half (Llama2, f32) | 236 B | 301 B | 1088 B |

The masked form is smaller because the body is recorded once instead of
twice. `store_interleaved3` is the exception: x86 has no predicated
interleaved store (`supports_masked_interleaved_access()` is false), so it
uses the counted path even under mask folding — SVE's `ST3` and RVV's
segment stores would take the predicate directly.

### Target capability queries

`vector_target` (`jit_kernel_target.hpp`) is the DSL's
`TargetTransformInfo`: `supports_masked_access(elem_bytes)`,
`supports_masked_interleaved_access()`, `preferred_tail_folding()`,
`predicate_pool()`. Selected from the host ISA at kernel construction.
Queries only — instruction emission stays in the arch's `jit_kernel`, so a
second architecture adds a sibling generator rather than editing this
interface.

### Type-converting load/store

`ir_load`/`ir_store` dispatch on the pointer element type:

| Type | Load | Store |
|---|---|---|
| `float` | `vmovups` | `vmovups` |
| `uint8_t` | `vpmovzxbd` + `vcvtdq2ps` | `vcvtps2dq` (own IR value) + `vpmovusdb` |
| `ov::float16` | `vcvtph2ps` | `vcvtps2ph` |
| `ov::bfloat16` | `vpmovzxwd` + `vpslld 16` | `vpsrld 16` (own IR value) + `vpmovdw` (truncating) |

Narrowing conversions are separate IR values rather than in-place edits of
the stored register: an op that clobbers a register it only declares as a
read corrupts the value for later uses. Both conversions have
non-destructive encodings, so the allocator still reuses the source
register when the source dies at the store.

### Semantic vector ops

`Insn2`: `vaddps vsubps vmulps vmaxps vminps`.
`Insn3`: `fmadd231ps fnmadd231ps fmsub231ps` (tied operand 0 — the seed
copy is coalesced away).
Plus `vec_copy`, `vec_permute` (table and address are IR values),
`shuffle`, `clamp`, `deinterleave2`/`interleave2`, `store_interleaved3`,
`ir_broadcast` (address or IR pointer), `ir_zero`, `ir_cmp`, `ir_if`.

## Kernels ported

### color_convert

NV12 and I420, `f32` and `u8`, one body per converter, whole kernel in IR.
`begin_ir()` precedes `arg()` so pointers, width and colorFormat are IR GPR
values; `make_ir_ptr` carries strides (`N`, `N/2` for subsampled planes,
`3N` for interleaved output); `foreach_with_epilogue` supplies the tail;
`ir_if(colorFormat)` selects RGB vs BGR store order. No `if constexpr`
split between f32 and u8 — the element type flows through the load/store
dispatch.

Dead leftovers from the eager era are still in the file:
`jit_uni_converter::yuv_to_rgb`, `JitConverter::load_yuv`,
`JitConverter::unpack_uv` have no call sites.

### RoPE (`OV_JIT_IR_ROPE=1`)

`rotary_half_ir` and `rotary_interleave_ir`, both on `foreach_vec`.
`rotary_half_ir` used to refuse `half_rotary_ndims % N != 0` because the
epilogue strategy expanded into ~20 GPR values and exhausted the pool;
under mask folding the tail is a predicated iteration with no extra GPRs,
so the guard is gone and all 19 `smoke_RoPETest*` cases pass in IR mode —
including the three `smoke_RoPETestQwenVL` shapes (half=40, N=16) that had
been failing since the kernel was written.

### Generated code (Llama2, rotary_ndims=128, f32, AVX-512)

| | Legacy | IR (no unroll) | IR (unroll=4) | Clang -O3 |
|---|--------|---------------|--------------|-----------|
| Code size | 1088 B | 227 B | 711 B | ~280 B |
| Data tables | 640 B | 0 | 0 | 0 |
| Vector insns/iter | 12 | 12 | 12 | 10* |
| Registers | 5 | 4 | 4 | 3 |

\* Clang folds loads into FMA memory operands.

Measured on the earlier `foreach_predicated` version of the kernel; the
current `foreach_with_epilogue` version has not been re-measured, and the
unroll column no longer has a mechanism behind it (see below).

## Open hazards and gaps

1. **No spiller.** Pool exhaustion is a hard failure. Needs frame sizing
   after allocation plus target hooks for store/load of a physical
   register. Blocks RoPE's QwenVL shapes.
2. **No dominance check.** The verifier accepts IR whose reads are not
   dominated by a def; liveness over-approximates instead.
3. ~~**Mask registers are not allocated.**~~ **Fixed** — `Mask` is a
   register class with a per-target allocation order, and predicates are
   ordinary IR values read by the ops that use them.
4. **No IR-level unrolling.** The pass was removed: the trip count lives
   inside the loop header's emit closure, so a pass over the IR cannot
   scale it to a cloned body. The old pass silently made kernels process
   `factor ×` the data — heap corruption on RoPE, wrong pixels on
   color_convert. Recording-time unrolling
   (`foreach_predicated(..., unroll)`) still works. An IR pass becomes
   possible once the loop bound and step are IR operands of a real loop
   op instead of captured immediates.
5. **The epilogue strategy records the body twice** — inherent to it, and
   now only chosen on targets without predication (AVX2, SSE, NEON). The
   masked strategy records once and is 36% smaller on the NV12 converter.
6. **No predicated interleaved store on x86.** `store_interleaved3` under
   a short active length still builds the interleave in a stack slot and
   copies `count*3` elements out. Three separately-derived masks would fix
   it on AVX-512; SVE (`ST3`) and RVV (segment stores) take the predicate
   directly. `supports_masked_interleaved_access()` is the switch.
7. **`length` (RVV) tail folding is declared, not implemented.** Needs a
   RISC-V generator and `vl` modelled as machine state with `vsetvli`
   insertion in the loop header. `foreach_vec` throws instead of
   approximating it.
8. **Only an x86-64 generator exists**, and the DSL has no arch boundary.
   See "Multi-architecture plan" below for the measurement, the three real
   gaps and the phasing.
9. **`bf16` store truncates** instead of rounding to nearest even
   (`vcvtneps2bf16` where available).
10. **GPR-hungry scalarized access.** `ir_load_partial` /
    `ir_store_partial` / `ir_memcpy` cost ~20 GPR values, which is why
    they are now only reached on targets without predication.
11. **Repo hygiene.** `jit_kernel.hpp` is 2662 lines; the design docs live
   in the repo root; the worktree carries `llvm-project/`,
   `dnnl_dump_*.bin`, `report_*.xml`, `rope_intrinsics.cpp`. Not
   upstreamable as a single change — wants splitting into IR core +
   tests / DSL / color_convert / RoPE.

## Multi-architecture plan

Targets: x86-64 (AVX2, AVX-512), AArch64 (NEON, SVE), RISC-V (RVV 1.0).
The IR was built arch-neutral on purpose; the DSL was not. Before planning,
the coupling was measured rather than assumed:

| Layer | Arch coupling | Reusable |
|---|---|---|
| `jit_kernel_ir.{hpp,cpp}` — IR, CFG/liveness, allocator, remat, verifier, passes | 0 xbyak references in the `.cpp`; 3 in the `.hpp`, all inside comments | as-is |
| `jit_kernel_target.{hpp,cpp}` — capability queries | none | as-is |
| `jit_kernel.{hpp,cpp}` — DSL + emit closures | 207 xbyak/x86 references across 63 emit closures | no |
| DSL type surface | 184 sites templated on `size_t N` / `float[N]` | blocks scalable vectors |
| Register tuples (consecutive registers) | not modelled anywhere | blocks SVE `ST3`, RVV segment ops |

### Three real gaps

1. **No arch boundary in the DSL.** `jit_kernel` derives from
   `dnnl::impl::cpu::x64::jit_generator_t` and mixes the portable DSL
   surface with 63 x86 emit closures in one header. `jit_kernel.md`'s
   Phase 3 boundary (`jit_kernel_base<ArchTraits>`) is still only a
   document.
2. **Compile-time `N`.** `variable<float[N]>` and
   `reg_traits<T[N]>` (byte size → register type) fix the vector width in
   the C++ type system. SVE and RVV are length-agnostic. Note what this
   is *not*: the allocator and live ranges are width-agnostic already
   (they reason about values, not bytes), and `vlen` is width-agnostic by
   construction. The blocker is the DSL's typing, and making it symbolic
   (LLVM's `vscale`) touches `variable`, `reg_traits`, `ir_ptr` stride
   arithmetic, `foreach_vec`'s `count >> log2(N)`, `ir_active_lane_mask<N>`,
   alloca sizing, `set_vec_width`, and the interleave/permute lowerings.
3. **Register tuple constraints.** SVE's `ST3W {z0-z2}, p0, [x]` requires
   *consecutive* registers, as do RVV segment loads/stores. LLVM models
   this with tuple register classes (`ZPR3`); we model nothing. Without
   it, `supports_masked_interleaved_access()` stays false on SVE too and
   `store_interleaved3` keeps its stack-slot fallback — losing the one
   operation where SVE would beat x86 outright.

### What already maps better on SVE than on x86

Evidence the abstraction is not merely x86 with different names:

- `ir_active_lane_mask` → `whilelt p0.s, x0, x1`: one instruction instead
  of `cmp/bts/dec/kmov`, and no early-clobber constraint needed.
- `RegisterClass::Mask` → `PPR` / `PPR_3b` allocation order; the class and
  its per-target order already exist.
- `tail_folding::mask` → SVE's native idiom; `foreach_vec` needs no change.
- `TwoAddressPass` → `FMLA` is destructive in its accumulator too;
  `MOVPRFX` is the non-destructive form, i.e. a lowering detail.
- Predicated access gives fault suppression architecturally, which is the
  justification for preferring masked tails in the first place.

### Phasing

**Phase A — prove separability on x86 (no cross toolchain).**
Extract the arch-neutral layer: move `vlen` out of the x86 `jit_kernel`,
split the DSL surface from the emit closures (`jit_kernel_base` +
`jit_kernel_x64`), and add a mock `vector_target` plus a recording-only
generator so the *decisions* can be tested without emitting anything.
Exit criteria: a test with a target reporting
`supports_masked_interleaved_access() == true` records a predicated
interleaved store instead of alloca + memcpy — a branch no x86 target can
reach today, hence currently untested; and `jit_kernel_ir` +
`jit_kernel_target` compile with no reference to the x64 generator.
This is the cheapest measurement of how much of gap 2 we actually need.

**Phase B — AArch64/NEON, fixed `N = 4`, one kernel.**
No predication, so the target reports `epilogue` folding — the path
already A/B-tested on x86 via `OV_JIT_TAIL_FOLDING`. Low intellectual
risk, high plumbing risk, which is the point: it shakes out the emit
closure boundary, `address_frame`, `stack_frame` and SP alignment,
oneDNN's `Xbyak_aarch64` naming, i-cache maintenance after code
generation, and CI under `qemu-aarch64`.
Exit criteria: the differential tests pass under emulation at NEON width.

**Phase C — SVE**, in this order: `whilelt` masks (trivial), predicated
arithmetic (`Insn2`/`Insn3` gain an optional predicate operand), tuple
register classes for `ST3` (gap 3), scalable `N` (gap 2 — a redesign, not
a port).

RVV follows the same staging and reuses Phase A/B work; the in-tree
`riscv64::jit_generator` (built on `xbyak_riscv`) is the generator to
layer on, and `tail_folding::length` is already declared for it.

### Verification without the hardware

- `qemu-aarch64` and `qemu-riscv64` 8.0.5 are installed here; `qemu-user-static`
  8.2.2 is packaged. RVV 1.0 is supported: `-cpu rv64,v=true,vlen=N` accepts
  `vext_spec=v1.0` and defaults to it. The old experimental `x-v` property
  is *rejected* by 8.0.5 — worth noting, because
  `.github/workflows/linux_riscv.yml` still uses
  `-cpu rv64,x-v=true,vlen=256` against the toolchain's bundled QEMU. Pin
  `vext_spec` explicitly when we start using it, or risk testing RVV 0.7.1
  semantics by accident.
- The repo already runs `ov_cpu_func_tests` under `qemu-riscv64` in CI, but
  not `ov_cpu_unit_tests` — adding the latter would put `JitKernelIR.*`
  (including the fuzz and differential tests) on the RISC-V gate cheaply.
- **Sweep the vector length.** One binary at `vlen=128/256/512` (RVV) or
  `-cpu max,sve128=on|sve256=on|sve512=on` (SVE) is the one thing
  emulation does better than hardware, and it is exactly what catches
  fixed-width assumptions.

### What emulation cannot establish

These fail deterministically on silicon and never under QEMU, so each
target needs one hardware run before we claim support:

1. **I-cache maintenance.** x86 keeps I-cache coherent with stores;
   AArch64 and RISC-V do not (`dc cvau`/`dsb`/`ic ivau`/`isb`, or
   `fence.i`). qemu-user invalidates its own translation blocks when guest
   code pages are written, so a JIT missing the required maintenance runs
   perfectly under emulation and executes stale bytes on hardware.
   `Xbyak_aarch64` and `xbyak_riscv` handle this — worth confirming once
   rather than assuming.
2. **Fault suppression on predicated access**, which is why masked tails
   are preferable to scalarized copies at all. Partial-fault behaviour on
   vector accesses is a weak spot in emulators; RVV first-fault loads
   (`vleff`) may legally return fewer elements on hardware than QEMU
   returns.
3. **Unspecified-value policies.** RVV `vta`/`vma` leave tail and inactive
   destination elements architecturally unspecified; QEMU picks one
   behaviour, hardware may pick another. Same for `vill` after an
   unsupported `vsetvli`.
4. **Feature gating and unaligned access.** `-cpu ...,v=true` grants RVV
   1.0 wholesale and `sve512=on` grants a width no shipping Neoverse core
   has; real chips ship subsets, and some RVV boards implement 0.7.1
   (T-Head C906/C910). Unaligned vector access is permitted by default in
   QEMU and optional in RISC-V hardware, while both our stack-bounce and
   masked paths assume it.

Cheapest hardware check per target: run `JitKernel*` plus the ConvertColor
and RoPE functional tests, with the differential test's buffers placed so
a tail access ends exactly at a page boundary and the next page is
unmapped. That single arrangement exercises i-cache maintenance, fault
suppression, feature gating and unaligned access at once — and it is worth
adding to the suite regardless, since it also hardens the x86 masked path.

## Environment variables

| Variable | Effect |
|---|---|
| `OV_JIT_IR_DUMP` | dump IR, live ranges and assignment before/after allocation |
| `OV_JIT_IR_TRACE` | trace recording, CFG/liveness, allocation and lowering |
| `OV_JIT_IR_ROPE` | select the IR RoPE kernel instead of the legacy one |
| `OV_JIT_TAIL_FOLDING` | `epilogue` / `mask` / `length` — override the target's tail-folding choice (LLVM's `-prefer-predicate-over-epilogue`) |

## Test status (2026-09-04, RelWithDebInfo, AVX-512 host)

Every suite run twice, once per tail-folding strategy
(`OV_JIT_TAIL_FOLDING=mask` and `=epilogue`), with identical results:

- `ov_cpu_unit_tests --gtest_filter='JitKernel*'`: **47/47 pass**
  (43 `JitKernelIR.*`, 4 `JitKernel.*`).
- `ov_cpu_func_tests --gtest_filter='smoke_TestsConvertColor*'`: **26/26
  pass**, including the `u8` accuracy case (144×16).
- `OV_JIT_IR_ROPE=1 ov_cpu_func_tests --gtest_filter='smoke_RoPETest*'`:
  **19/19 pass** (was 16/19 before the tail-folding work).
- `ov_cpu_func_tests --gtest_filter='smoke_ConvertCPULayerTest_4D_Static/*'`:
  32 pass / 68 skipped (unchanged; covers the eager `cpu_convert` path).

Coverage highlights: live-range construction across loops and branches,
loop-carried liveness around the back edge, loop-invariant reads,
branch-local disjointness, tied-operand coalescing, GPR/Vec mixing,
rematerialization, verifier rejection cases, a 400-seed randomized
allocation fuzz test ("every successful allocation satisfies the
verifier"), a differential FMA-with-epilogue kernel over 27 widths
(exercises partial load *and* partial store), a narrowing-store
clobber test, and an eager-mode `foreach` + `stack_frame::clear` test.

## Review pass of 2026-09-03 — what changed

Applied in this order, each verified against the three suites above:

1. **Spill stub removed.** It created spill/reload ops with empty emit
   closures, no lowering support and no frame space; if it ever triggered
   the kernel was silently wrong. Exhaustion now throws with a diagnostic.
2. **Verifier + tests added** (see above), plus a **partial store**:
   `ir_store` had no partial path, so `foreach_with_epilogue` with a plain
   store overran the destination by up to N-1 elements. It was unexploited
   only because the one epilogue kernel stored through `store_interleaved3`.
3. **Liveness replaced** with CFG construction plus a dataflow fixpoint,
   deleting the two extension heuristics. Four historical correctness bugs
   shared that single root cause.
4. **`linear_scan` renamed to `assign_registers`** and documented as what
   it is.
5. **Remat env knobs and the legacy `rematerialize_for_pressure` pre-pass
   removed** — the pre-pass had no production caller and its four
   `OV_JIT_IR_DISABLE_*` / `_SINGLE_USE` switches gated correctness.
6. **Loop-unrolling pass removed** after it was shown to corrupt the heap
   on RoPE and produce wrong pixels on color_convert when enabled.
7. **Vector pool fixed**: it had been filled from the GPR index range
   (16 entries) and did not exclude eagerly reserved registers. Now an
   explicit allocation order, extended to 32 registers for kernels that
   declare 512-bit width.
8. **Narrowing stores no longer clobber their source** (conversion is its
   own IR value).
9. **Scratch registers modelled as IR values** — the `push(param1)` in the
   permute-table load and the `push` in the predicated mask setup are gone.
10. **Captured-physical-register fallbacks deleted** from
    `ir_load`/`ir_store`/`ir_advance`/`foreach_predicated`; pointers must
    be IR values, and a new `ir_broadcast(pointer, offset)` covers the
    case that needed the fallback. Tests were converted accordingly.
11. **Eager `foreach` restored.** Dropping eager mode had left `foreach`
    dereferencing a null IR, which would have crashed `cpu_convert.cpp`'s
    `jit_convert_array` (and `stack_frame::clear`) on the first f16/bf16/f8
    conversion. Covered by a new test.
12. **Ambient tail state removed** in favour of the explicit `vlen`
    argument; `OpKind` (which only existed for the epilogue pass that
    `vlen` makes unnecessary) removed with it.

Net: the IR core is smaller (1500 vs 1704 lines) despite gaining a CFG,
dataflow liveness and a verifier.

## Multi-arch pass of 2026-09-04 — active length, predicates, targets

The previous pass made the active length explicit but left it x86-shaped:
`vlen` held an `Xbyak::Opmask`, and one of its three states *was* the
stack-bounce mechanism. Reworked along LLVM's lines, with AVX2/AVX-512,
NEON/SVE and RVV as the intended targets:

1. **`vlen` is semantics only** — a mask value plus an element count,
   either optional, neither naming a register type. Same shape as LLVM's
   vector-predication operands (`<n x i1>` mask + `i32` evl), for the same
   reason: the front end states what is active, the target decides how.
2. **`Mask` is a register class** with a per-target allocation order.
   `Opmask(1)` is gone; predicates are IR values read by the ops that use
   them, so several accesses can share one and the allocator places them.
3. **`vector_target`** answers the questions the front end must ask before
   recording — masked access legality, masked interleaved access,
   preferred tail folding, predicate pool. The DSL's
   `TargetTransformInfo`.
4. **`foreach_vec`** is the one loop kernels write; the target picks
   `mask` / `epilogue` / `length`, exactly as LLVM's vectorizer picks a
   `TailFoldingStyle`. `OV_JIT_TAIL_FOLDING` overrides it, which is also
   how the epilogue path stays tested on an AVX-512 host.
5. **Scalarized access is a fallback, not a mode.** A count with no
   predicate is turned into a predicate when the target supports it, and
   only otherwise goes through the stack slot — the role LLVM gives
   `ScalarizeMaskedMemIntrin`.
6. **Early-clobber defs** were added because the mask computation needed
   them, and the differential test proved it: the allocator had given the
   def its dying read's register, which the multi-instruction sequence
   then clobbered before use.
7. **RoPE's shape restriction is gone** — predicated tails cost no extra
   GPRs, so `half_rotary_ndims % N != 0` works and the QwenVL cases pass.

What is genuinely portable now: the loop construct, the active-length
model, the register classes with per-target orders, and the capability
queries. What still blocks non-x86 targets: there is only an x86-64
generator, and scalable vectors (SVE, RVV) need `N` to stop being a
compile-time constant — a `variable<float[N]>` change, not a `vlen` one.

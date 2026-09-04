# Register allocation for `jit_kernel`

Design notes for the lightweight register allocator that will run as an
opt-in IR mode on top of `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel.hpp`.
Companion to `jit_kernel.md` — specifically the "Register allocation and
pressure analysis" section, which motivates why this is needed.

Scope was deliberately narrow: **this is a register allocator, not a
compiler.** No CSE, no LICM, no constant folding, no instruction selection,
no pattern matching.

**Status (2026-09-03).** The narrow scope holds for the pipeline as it
stands: two-address lowering, liveness, allocation with
rematerialization, verification, lowering. It did not hold in between — a
loop-unrolling pass and a spill stub were added and have since been
removed, the former because an IR pass cannot scale a trip count that
lives inside an emit closure, the latter because it emitted no code at
all. What is no longer literally true is "one DSL call = one emitted
instruction" (type conversion, interleaved stores and partial access all
expand), and eager mode for *vector* work was removed rather than kept as
the default.

This document is updated in place: design intent is retained where it
still describes the code, and sections the implementation diverged from
say what the code does instead. `jit_kernel_journal.md` carries the
current status, the open hazards and the change log.

## Why

The scope-based lifetime tracking that today's `shared_reg` provides has
two load-bearing gaps, quantified on the color_convert refactor
(`jit_kernel.md`, "Empirical evidence" subsection):

1. **C++ scope ≠ last use.** A `variable` handle holds a register until
   its enclosing C++ scope ends, not until the last instruction that
   reads the value. Color_convert measurement showed ~6 vector registers
   dead-held at any point inside `yuv_to_rgb`, tipping peak pressure over
   the 16-register cap and blocking loop-invariant hoisting.
2. **No interval splitting.** Even with perfect last-use tracking, a
   scope-based allocator cannot spill a value to stack for a cold region
   and reload it when next needed. Manual hoisting of yuv_to_rgb
   coefficients produced peak pressure of ~19 live because the hoisted
   values sat live across the entire loop body alongside transients.

Both gaps are things a register allocator handles naturally, and neither is
something the author can easily manage by hand — reasoning about global
liveness while writing locally readable code is the one thing the DSL
cannot leave to the author.

Gap 2 (interval splitting/spilling) is still *not* closed: there is no
spiller, so a kernel that exceeds its register file fails to build rather
than spilling. What closed was gap 1, plus rematerialization for values
that are cheaper to recompute than to keep.

## Relationship to by-value variable semantics

The allocator depends on one DSL-side invariant: **SSA at the IR level.**
Every value-producing op writes to a fresh value id. `operator=` rebinds
the handle to a new id and never mutates an existing value in place.

The `variable` by-value refactor (see `jit_kernel.md`, "Value-transforming
operations are by value") makes this natural. Every arithmetic, shuffle,
permute, and cast already returns a fresh `variable`. The IR records the
fresh id each time. Any future optimization (remat, splitting, coalescing
beyond trivial) assumes SSA — if we ever broke this invariant, all
extensions would break with it.

Passing variables by value through stage boundaries is consistent with the
same model: each by-value parameter is recorded as a new "use" in the IR's
op list. Handle drops become irrelevant in IR mode — liveness lives in
the op graph, not in C++ scope. In eager mode by-value costs a refcount
bump per call (compile-time of kernel generator, not generated code), which
is negligible.

## Two modes, one DSL — superseded

Original plan: eager mode stays the default, IR mode is opt-in per kernel.

**As implemented:** there is one mode for vector work. `vec_op`,
`vec_copy`, `vec_permute`, `ir_load`, `ir_store` and friends dereference
`_ir` unconditionally and only work between `begin_ir()` and `end_ir()`.
The eager machinery that remains is scalar/GPR only — `var<size_t>`,
`foreach`, `_if/_then/_else`, `stack_frame`, `jit_load_emitter` interop —
and is still used by `cpu_convert.cpp`, which never touches vector
`variable`s.

Why the split collapsed: every DSL vector primitive needs the allocator to
see its operands. Keeping a parallel eager implementation meant two
lowering paths per primitive, and the eager path is exactly where the
"captured physical register index" bugs came from. Those fallbacks are now
deleted — pointers handed to `ir_load`/`ir_store`/`ir_advance` must be IR
values, i.e. `arg()` is called inside IR mode.

Eager **control flow** did have to stay: `foreach` and
`stack_frame::clear()` are used by `cpu_convert.cpp`, which emits raw
xbyak. Deleting the eager `foreach` left it dereferencing a null IR — a
latent crash on the first f16/bf16/f8 conversion — so it is back, with a
test.

Consequence for kernel authors: adopting the vector DSL means adopting IR
mode. The "measure first, escalate to IR mode only under pressure"
gating no longer applies.

## Instructions as data

The core design problem: in the existing `jit_kernel`, instructions are
**methods** (`uni_vaddps`, `uni_vmulps`, …). Methods can't be passed
around, enumerated, or generically wrapped in C++. Wrapping each method
for IR mode would require per-instruction boilerplate — a macro or
manual wrapper for every `uni_*` call and every FMA variant.

The solution: **make instructions data, not methods.** An instruction
is an enum value. A single generic dispatch function per arity handles
the IR-vs-eager branch. A single `lower()` function maps enum values to
xbyak calls. No per-instruction wrapper code exists anywhere.

### Instruction enums

```cpp
enum class Insn2 : uint8_t { vaddps, vmulps, vsubps, vdivps, vandps, vorps, vxorps, COUNT };
enum class Insn3 : uint8_t { fmadd213ps, fmadd231ps, fnmadd231ps, COUNT };
```

Adding a new instruction: add an enum value + one `case` line in
`lower()`. Nothing else changes.

### Generic dispatch

One `vec_op()` function per arity. Records the op and captures the
instruction enum in the emit closure — **written once, covers all
instructions of that arity**. As implemented (`jit_kernel.hpp:1567`),
there is no mode switch, because there is no eager path:

```cpp
template <size_t N>
variable<float[N]> jit_kernel::vec_op(Insn2 insn,
                                      const variable<float[N]>& a,
                                      const variable<float[N]>& b) {
    using reg_type = typename reg_traits<float[N]>::type;
    auto vid = _ir->def({a.vid(), b.vid()},
        [this, insn](const jit_kernel_ir::EmitContext& ctx) {
            lower(insn, reg_type(ctx.def->idx),
                        reg_type(ctx.reads[0].idx),
                        reg_type(ctx.reads[1].idx));
        }, "vec_op");
    return variable<float[N]>(*this, vid);
}
```

The ternary form (`vec_op(Insn3, seed, a, b)`) records `def_tied(..., 0)`
so the seed is a tied operand; `TwoAddressPass` then inserts the COPY and
the coalescer usually removes it.

Width dispatch lives inside `ctx.regs()` — one place, all arities. The
`lower()` lambda uses `auto`-typed register parameters, so `uni_*`
overloads resolve naturally from whatever width `regs()` produces.

### Lowering switch

The only place that mentions xbyak instruction names:

```cpp
void lower(Insn2 insn, const auto& d, const auto& s1, const auto& s2) {
    switch (insn) {
        case Insn2::vaddps: uni_vaddps(d, s1, s2); break;
        case Insn2::vmulps: uni_vmulps(d, s1, s2); break;
        case Insn2::vsubps: uni_vsubps(d, s1, s2); break;
        // ...
    }
}
```

FMA lowering handles the in-place mutation that xbyak requires:

```cpp
void lower(Insn3 insn, const auto& d, const auto& s1, const auto& s2, const auto& s3) {
    if (d != s1) uni_vmovups(d, s1);  // value semantics: copy before mutate
    switch (insn) {
        case Insn3::fmadd213ps: uni_vfmadd213ps(d, s2, s3); break;
        case Insn3::fmadd231ps: uni_vfmadd231ps(d, s2, s3); break;
        // ...
    }
}
```

The copy-before-mutate is eliminable by the allocator's coalescing pass
(Slice 4, `is_copy` flag) when the source interval ends at the FMA.

### Syntactic sugar

Lightweight callable members give method-call syntax without
per-instruction logic:

```cpp
struct Op2 {
    jit_kernel* self;
    Insn2 insn;
    variable operator()(const variable& a, const variable& b) const {
        return self->op(insn, a, b);
    }
};

// In jit_kernel — one line per instruction, zero logic:
Op2 vaddps{this, Insn2::vaddps};
Op2 vmulps{this, Insn2::vmulps};
Op3 fma213{this, Insn3::fmadd213ps};
```

Kernel code reads naturally:

```cpp
auto c = vaddps(a, b);
auto d = fma213(a, b, c);
```

### Complex operations

Operations with non-uniform structure (`store_interleaved3`, custom
shuffle sequences) do not fit any arity enum. They are written manually
using `ir_.def()` / `ir_.use()` directly — the same primitives that
`op()` uses internally. These are the minority (~5-10 across the whole
kernel) and each has unique logic that deserves an explicit body.

### Runtime cost

- **Kernel build time:** record + analyse + allocate + lower. The
  allocator is the dominant cost and is quadratic-ish in pathological
  cases: interference checks are O(values × regs × segments), and every
  remat or spill insertion restarts live-range analysis from scratch. Real
  kernels are tens of ops, so this is microseconds — but it is not the
  "~2× of eager" the original estimate assumed.
- **Generated code:** unaffected by the mechanism; the point of the
  exercise is that the register assignment is better than scope-based
  allocation could produce.
- **C++ compilation:** `jit_kernel.hpp` is now 2521 lines of templates and
  every emit site is a `std::function`. This is the real compile-time
  cost, and it is why the header wants splitting.

## IR shape

Minimal. As implemented (`jit_kernel_ir.hpp:83`), control flow is
**structural nesting** (MLIR-style regions), not a marker stream:

```cpp
using value_id = uint32_t;

struct Op {
    std::vector<value_id> reads;
    value_id def = invalid_value;        // invalid_value = no def (e.g. store)
    EmitFn emit;                         // captures insn + lowering call
    bool is_copy = false;                // trivial coalescing hint
    int tied_to = -1;                    // LLVM tied operand: index into reads[]
    RegisterClass def_rc = Vec;          // Vec | GPR
    OpKind kind = Generic;               // Generic | Load | Store (transform passes)
    std::unique_ptr<IR> body;            // non-null = region op (loop, branch)
    bool is_loop = false;                // extend ranges across the body
    const char* name = "";               // debug tag
};

class IR {
    std::list<Op> _ops;
    std::list<Op>* _cursor = nullptr;    // non-null = recording into a body
    value_id _next_value = 0;
};
```

The `CFGMarker` variant from the original sketch was never built. Regions
won because the DSL already nests (`foreach` and `ir_if` take body
lambdas), so recording a nested `IR` costs a cursor save/restore and
keeps loop structure explicit.

The cost of that choice shows up in analysis: with a marker stream you get
basic blocks for free and can run textbook dataflow over them. With
regions you either build the CFG on demand or approximate liveness by
walking the tree — the implementation currently does the latter, which is
hazard #2 in `jit_kernel_journal.md`.

**No op taxonomy at the IR level.** The allocator never inspects which
instruction an `Op` represents — it only reads the dependency shape
(reads, def) and calls the emit closure at lowering time. The `Insn2` /
`Insn3` enums live at the DSL wrapper level; the IR is instruction-
agnostic.

The `emit` closure is built by `op()` and captures the instruction enum
plus a pointer to `lower()`. Adding a new DSL primitive requires no
changes to the IR, the allocator, or the lowering pass — only a new
enum value and a case in the appropriate `lower()` switch.

**CFG markers live inline in the stream**, not as a separate block graph.
This is the minimum representation that carries enough structure for
liveness dataflow and loop handling, and it trades cheaply against
building a real CFG later if LICM or loop unrolling ever becomes
interesting (see "Extensibility" below).

## The allocator — three passes

### Pass 1: Liveness dataflow — **implemented (2026-09-03)**

Backward pass over basic blocks computing per-block `live_in` / `live_out`
sets, iterated to a fixpoint so back edges converge.

The CFG is built from the region tree: a loop region contributes a
preheader → header edge, header → {body entry, loop exit}, and latch →
header (the back edge); a conditional region contributes header → {body
entry, join} and body exit → join. Loops need the separate header block
because the back edge targets the compare, not the counter initialisation
— folding them together makes values that live across a loop appear dead
inside it. Region ops own an index of their own, so the header's reads
(counter, bound) are live where the compare reads them.

Segments come from block-level liveness: values live-in open at the block
start, defs open at their late slot, and each open segment closes at the
block end (live-out) or at its last read. Blocks are created in walk
order, so consecutive blocks are adjacent in slot space and `addSegment`
merges pass-through liveness into single segments.

This replaced a per-nesting-level tree walk with two extension heuristics.
Four historical correctness bugs shared that root cause.

Standard dataflow equations:
```
live_out[B] = ∪ live_in[S]  for each successor S of B
live_in[B]  = (live_out[B] − def[B]) ∪ use[B]
```

Basic blocks are derived from `CFGMarker` positions in the stream. No
explicit block graph is built — block boundaries are start-of-stream,
after every `Branch` or `Jmp`, and before every `Label`. Block successors
are inferred from the marker kind (fall-through vs jump target).

~80 lines. The one non-trivial piece of the minimum. Straight-line code
(no branches, no loops) skips this entirely — intervals come from scope
scanning — but `foreach` and `_if/_then/_else` force it.

### Pass 2: Interval construction

Per value id, compute `[first_def, last_use]` extended across blocks per
the liveness sets. Values live through a loop body have intervals
covering the entire loop from `LoopBegin` to `LoopEnd` plus any uses
after. Values live across an `_if/_then/_else` merge have intervals that
span both branches.

~60 lines.

### Pass 3: Linear scan

Classical Poletto & Sarkar 1999. Walk intervals sorted by `start`:

```
active = {}   // currently allocated intervals
for each interval I in order of I.start:
    expire(active, I.start)       // release regs whose intervals ended
    if free regs available:
        assign I a free reg
    else:
        victim = pick_victim(active ∪ {I}, I.start)
        spill(victim)
        assign I the victim's reg (or its slot if I itself was chosen)
```

`pick_victim` is a swappable function. Minimum implementation:
**furthest-next-use** — spill the interval whose next read is furthest
in the future. `expire` moves ended intervals out of `active` and frees
their physical regs.

**Spill mechanics**: allocator inserts a `Store` op at the victim's
current position (dump to a stack slot) and a `Load` op before the
victim's next read (reload from the slot). The inserted ops are regular
`Op` entries with their own `emit` closures — they flow through the
lowering pass like any other op.

**Trivial coalescing**: when processing a `Copy` op where the source's
interval ends at the copy point and the def's interval starts, reuse the
source's physical register for the def. One-line check inside the main
loop, no separate pass.

**Three independent pools** (vector regs, GPRs, mask/k-regs). Allocate
separately. No unified constraint solver — each pool runs the same
linear-scan logic over its own subset of ops and intervals.

~150 lines for the scan core. ~100 lines for spill/reload mechanics
including stack slot management.

### Pass 4: Lowering

Walk the IR stream in order. For each `Op`, call
`emit(EmitContext{...})` with the physical registers the allocator
assigned. The emit closure — built by `op()` at recording time —
calls `lower(insn, regs...)` which maps the instruction enum to the
corresponding xbyak call. For each `CFGMarker`, emit the corresponding
label or jump directly through `jit_generator_t`.

**No allocator logic in this pass.** Pure mechanical substitution. The
`lower()` switch is the only code that names specific xbyak instructions.
This separation means any future allocator can plug into the same
lowering pass without changes, and any new instruction only touches the
enum + switch — not the allocator, not the IR, not the lowering loop.

~100 lines.

## Current implementation status

The implementation has evolved significantly from the initial
"minimum" sketch. The architecture now follows LLVM conventions
where practical.

### Pass pipeline

`PassManager` + `PassContext` + `IRPass` (`jit_kernel_ir.hpp:353-465`).
Default pipeline: `LoopUnroll → TwoAddress → LiveRangeAnalysis → Dump →
RegisterAllocator → Dump → Lowering`. `PassContext` carries pool sizes,
analysis results, the spill slot table, the lowering callback and the
debug flags. Passes report `modified`, but the manager ignores it — there
is no invalidation model, so a transform pass must leave analyses in a
state the next pass can consume (currently guaranteed only by pipeline
order).

### Register classes and two pools

`RegisterClass::{Vec, GPR}` lives on the *virtual* register
(`Op::def_rc`, `LiveRange::rc`), not on `PhysReg` — the LLVM
`MachineRegisterInfo` split. One unified work list, three allocation orders
(`vec_pool_indices`, `gpr_pool_indices`, `mask_pool_indices`), all explicit
lists in LLVM's `AllocationOrder` sense. Neither is contiguous: GPR excludes
`rsp`/`rbp`/`abi_param1`, and both exclude anything the kernel reserved
eagerly (`arg()`, `reserve<>()`, the constants-table register) — which
also closed a hole where the allocator could hand out a register the
kernel was already holding.

**Scalable vector length does not reach the allocator.** Worth stating
because it is the usual assumption: live ranges are computed over values
and slot indices, never over byte widths, and the allocation orders are
lists of physical register indices. Nothing here changes if a vector
register's size is only known at run time. Vector *length* enters the
picture in two other places — the DSL's type surface
(`variable<float[N]>`, `reg_traits<T[N]>`, stride and alloca arithmetic)
and, on RVV, the machine state that `vsetvli` establishes. So SVE/RVV
scalability is a DSL question and a lowering question, not an allocator
question.

**Masks are the third class.** `RegisterClass::Mask` covers the predicate
file, with the per-target constraints in its allocation order rather than
in the allocator: x86 lists `k1..k7` (`k0` cannot encode a write-mask —
LLVM's `VK*WM`), SVE would list `p0..p7` (`PPR_3b`), RVV only `v0`
(`VMV0`). An ISA with no predicates returns an empty order and requesting
a mask throws. `vector_target::predicate_pool()` supplies it.

**The vector file needs a second distinction.** `zmm16..zmm31` exist on
AVX-512 but are EVEX-only, so VEX/SSE-encoded instructions cannot name
them. LLVM models this with separate register classes (VR128 vs VR128X);
here it is one opt-in per kernel: `set_vec_width(512)` extends the IR
allocation order to 32 registers, anything else keeps the low 16, and the
eager pool is always the low 16. Filling the pool from the GPR index range
(the original bug) hid half the AVX-512 file; handing high registers to
eager code produced Xbyak "not supported".

GPRs became IR-managed in commit `0ac8064ff2`, which retired the earlier
constraint "GPRs are not IR-managed; emit closures must not call
`var<>()`". Pointers, loop counters, partial counts and `ir_alloca`
addresses are all allocator-assigned values now.

### TwoAddressPass

`x86` FMA is destructive. Recording uses `def_tied(reads, 0, ...)`;
`TwoAddressPass` (`jit_kernel_ir.cpp:1125`) then rewrites

```
%r = FMA(%seed, %a, %b)     [tied_to=0]
```
into
```
%r = COPY(%seed)
%r = FMA(%r,   %a, %b)      [tie discharged]
```

i.e. it deliberately breaks SSA the way LLVM does after two-address
lowering: one value id, two defs, two `VNInfo`s. The coalescer in the
allocator gives `%r` the seed's register whenever the seed's range allows,
and lowering skips copies whose def and source agree — so the seed move
disappears in the common case.

### Spill — **removed; exhaustion throws**

There used to be a spill fallback whose `spill`/`reload` ops carried empty
emit closures, with no lowering support and no frame space
(`PassContext::spill_slots` offsets were never assigned, because the frame
is sized before the pipeline runs). It emitted nothing, so any kernel that
reached it was silently wrong. It is gone: `assign_registers` throws
`allocation_failure` naming the class, pool size, value, slot and the
number of interfering live values.

A real spiller needs two things this layer does not have yet:
1. **Frame sizing after allocation.** Spill slots are discovered during
   allocation, but `end_ir()` emits `sub rsp, total` before running the
   pipeline. Prologue insertion has to move behind the allocator (LLVM's
   PEI ordering).
2. **Target hooks.** Emitting a store/load of a *physical* register is
   target knowledge the IR layer does not own — the equivalent of
   `TargetInstrInfo::{storeRegToStackSlot,loadRegFromStackSlot}` plus
   `MachineFrameInfo::CreateSpillStackObject`.

Until then the honest failure mode is a diagnostic, and the kernel author
reduces pressure.

### Loop unrolling — **removed**

There was a `LoopUnrollPass` behind `OV_JIT_IR_UNROLL`. Enabling it
corrupted the heap on RoPE and produced wrong pixels on color_convert,
because the trip count lives inside the loop header's emit closure: a pass
over the IR can clone the body but cannot scale the count, so the kernel
processed `factor ×` the data. Its cloner also could not clone nested
region bodies, which silently dropped `ir_if` bodies.

Unrolling therefore stays at recording time, where the DSL still knows the
bound (`foreach_predicated`'s `unroll` argument). An IR-level pass becomes
possible once the loop bound and step are IR operands of a real loop op
instead of captured immediates — that is the prerequisite, not a patch.

### Verification — **implemented (2026-09-03)**

`verify()` runs as `VerifyPass` between allocation and lowering, always
(it is microseconds at kernel scale). It rejects malformed live ranges,
reads of undefined values, values referenced but unassigned, registers
outside their pool, leftover tied operands, and — the property that
matters — two values sharing a physical register with overlapping ranges.

Backed by rejection tests, a 400-seed randomized allocation fuzz test
("every successful allocation satisfies the verifier"), and a differential
kernel test against a scalar reference over 27 widths.

Still unchecked: **dominance**. A value defined only inside a conditional
region and used after it is invalid IR that passes verification; liveness
over-approximates its range instead of leaving a hole, so the outcome is
conservative rather than wrong.

### Live ranges (LLVM naming)

The old flat `Interval` (`[start, end]` pair per value) has been
replaced with **`Segment` + `LiveRange`** (LLVM naming):

```cpp
struct Segment { uint32_t start, end; };
struct LiveRange {
    value_id id;
    std::vector<Segment> segments;  // sorted, non-overlapping
    // ...
    bool liveAt(uint32_t index) const;
    void addSegment(Segment s);
};
```

`compute_live_ranges()` uses a **per-recursion-level local map**:
each call to the recursive walker tracks `value_id → {first, last}`
for values seen at that nesting level. On return, local segments are
flushed into the `LiveRange` via `addSegment()`. Sibling branch bodies
(separate recursive calls) naturally produce separate segments.

**Merge policy**: `addSegment()` merges overlapping segments but NOT
merely adjacent ones — this preserves branch boundary information.
`[0, 2]` and `[3, 5]` stay separate (branch gap), while `[0, 3]`
and `[2, 5]` merge to `[0, 5]`.

**Parent extension**: when a value is defined before a region (loop
or branch) and used inside, the parent's local segment is extended
to cover through the child's committed segments. For loops, the
segment is extended to the full loop body end. This prevents holes
in straight-line code between a def and its branch-body use.

### Allocator: LLVM-style per-register interference

The allocator uses **per-register segment unions** (LLVM approach).
Each physical register tracks a sorted list of all segments assigned
to it. A LiveRange can use a register iff none of its segments
overlap any segment already on that register.

```
for each LiveRange (sorted by beginIndex):
    for each PhysReg:
        if no segment overlap → assign, done
    if no register found → try remat, or throw
```

Two values with non-overlapping segments (e.g., intermediates in
different branches of an `ir_if`) naturally share a register. No
expire logic, no temp pools, no re-acquisition.

### Integrated rematerialization

Remat is integrated into the allocator (LLVM approach), not a
separate pre-pass. When the allocator cannot find a register for a
LiveRange:

1. Build a def map (`value_id → Op*`).
2. Among all assigned values whose segments interfere with the
   failing value, find the best rematerializable victim.
3. A value is **rematerializable** if:
   - its def Op has no reads (constant/broadcast), **OR**
   - all of its def Op's reads have ranges that span the victim's
     entire lifetime (input-aware remat — the clone's inputs are
     already live, adding no pressure).
4. Clone the victim at every use site via `remat_all_uses_impl()`.
   Each clone carries the victim's `reads` and `emit` closure.
5. Return `std::nullopt` — the caller recomputes live ranges and
   retries allocation.
6. If no rematerializable victim exists, throw `allocation_failure`.

The retry loop in `end_ir()` is bounded by the initial value count
to prevent infinite remat chains.

### Mechanical lowering

Unchanged: walk the IR tree, call `emit(EmitContext{def_reg, read_regs})`
for each Op. Region ops emit their header closure, then recurse into
the body.

### Tail handling — one construct, the target picks

**Superseded by `foreach_vec`.** Kernels call `foreach_vec<N>(count,
body)`; the body receives the iteration's active length (`vlen`) and hands
it to its memory operations. `vector_target::preferred_tail_folding()`
then selects the strategy — `mask` (one predicated loop: AVX-512, SVE),
`epilogue` (full-width loop plus counted tail: AVX2, SSE, NEON) or
`length` (per-iteration `vl`: RVV, declared and unimplemented). Exactly
LLVM's `TailFoldingStyle` selection, and `OV_JIT_TAIL_FOLDING` overrides
it the way `-prefer-predicate-over-epilogue` does.

Measured on AVX-512, both strategies passing all suites: the NV12 f32
converter is 833 B under `mask` versus 1309 B under `epilogue`, and RoPE
half is 236 B versus 301 B — the masked form records the body once.

The description of the two strategies below is retained because they are
what the two styles lower to.

### Tail handling — the two strategies

`foreach_with_epilogue<N>(width, body)` is what the production kernels
(color_convert, RoPE) use today. It records `body` **twice** — once in
`LoopMode::Full` inside `foreach(0, width/N)`, once in
`LoopMode::Partial` under `ir_if(tail != 0)` — and DSL methods
(`ir_load`, `store_interleaved3`, `ir_advance`) branch on the ambient
mode. Partial loads go through `ir_load_partial`: `ir_alloca` a slot, zero
it with a vector store loop, scalar-copy `count` elements, then load full
width. Portable across ISAs, but it costs a stack bounce and expands into
~20 GPR values, which is enough to exhaust the GPR pool (this is why
`rotary_half_ir` currently refuses non-multiple-of-N shapes).

`foreach_predicated<N>` is the older, AVX-512-only strategy, still used by
unit tests. It is the better shape and the one the numbers in
`jit_kernel_journal.md` were measured on; it just needs an AVX2 fallback.
Mask registers were not allocator-managed when this was written —
`Opmask(1)` was hardcoded and the mask setup borrowed a GPR via push/pop.
Both are fixed: predicates are `RegisterClass::Mask` values and the
scratch GPR is an early-clobber IR value.

The intended end state is one recording of the body plus an IR pass that
clones the loop and rewrites `OpKind::Load`/`Store` ops into partial or
masked variants — the `OpKind` tag exists for exactly that pass, which
has not been written yet.

Description of the predicated strategy follows.

Tail handling is not a separate code path — it's part of the loop.
`foreach_predicated<N>` emits a single loop that processes all elements
including the remainder:

1. Iteration count = `ceil(total / N)`.
2. Per-iteration mask setup (via `ir_use` closure, emitted at lowering
   time): `remaining >= N → kxnorw (all ones)`, else `(1 << remaining) - 1`.
3. Body executes with the mask. `ir_load` / `ir_store` automatically
   use it.
4. `remaining -= N` at the end of each iteration.

**Ambient predication**: `foreach_predicated` sets `_predicated = true`
and `_active_mask = k1` on the kernel before calling the body builder.
`ir_load` and `ir_store` check `_predicated` and delegate to masked
variants when set. Wrappers like `store_interleaved3` — which call
`ir_store` internally — gain masking for free with zero code changes.

This matches the universal vectorization pattern across ISAs:
- **AVX-512**: one loop, k-register mask. Zero overhead for full iterations.
- **ARM SVE**: one loop, predicate register via `whilelt`.
- **RISC-V V**: one loop, `vsetvl` configures element count.
- **AVX2** (future): recording-time decision to emit main loop (unmasked)
  + masked tail from the same body builder.

### Type-converting load/store

`ir_load` and `ir_store` dispatch on the pointer element type via
`if constexpr`. Adding a new type conversion = one `if constexpr`
branch in each function. No new API surface, no wrapper changes.

| Source → Dest | Load instructions | Store instructions |
|---|---|---|
| `float*` → `float[N]` | `vmovups` | `vmovups` |
| `uint8_t*` → `float[N]` | `vpmovzxbd` + `vcvtdq2ps` | `vcvtps2dq` + `vpmovusdb` |
| `bfloat16*` → `float[N]` (future) | `vpmovzxwd` + `vpslld 16` | `vcvtneps2bf16` |
| `float16*` → `float[N]` (future) | `vcvtph2ps` | `vcvtps2ph` |

All variants support k-register masking natively on AVX-512. The masked
paths follow the same `if constexpr` dispatch — no separate `_masked`
overloads in the public API.

This eliminates the need for `jit_load_emitter` / `jit_store_emitter`
for these types. Two direct instructions vs the general-purpose emitter
infrastructure.

### Legacy pre-pass

The old `rematerialize_for_pressure()` function remains available for
backward compatibility and tests, but is no longer called from
`end_ir()`. The allocator handles all remat decisions.

## The one ugly corner: escape-hatch barriers — not implemented

**Status:** never built, and currently not needed. The DSL-native
load/store path covers f32, u8, f16 and bf16 (the last two truncating on
store), so no production IR kernel calls a legacy emitter. `Op` has no
`clobbers` field. The section below is retained as the design to follow
if a kernel ever needs a type pair the DSL does not own.


Call-outs to legacy emitters (`jit_load_emitter`, `jit_store_emitter`,
injectors) are the part of the minimum that isn't minimal. Their
semantics from the allocator's point of view:

- Every caller-saved physical register is potentially clobbered.
- Any value live across the call that isn't in a callee-saved slot must
  be spilled before the call and reloaded after.

Recording shape: a barrier `Op` carries a `clobbers` bitmask marking the
registers it may destroy. The allocator treats every live value whose
current reg is in `clobbers` as requiring a spill/reload cycle around
the barrier — same machinery as normal pressure spills, just forced.

~50 lines. Ugly but unavoidable if IR mode must coexist with the legacy
emitter stack.

**Alternative that sidesteps this**: DSL-native load/store primitives for
f32 and u8 (described in `jit_kernel.md`, "IR-mode kernels and the
load/store split"). If color_convert's load/store paths are rewritten as
DSL-native primitives first, the barrier mechanism is not needed for
color_convert at all — the only primitives the kernel calls are
allocator-aware. The ~250 lines of DSL load/store is then a direct
prerequisite of IR mode adoption, not a separable perf upgrade.

Recommended ordering: ship DSL-native load/store for f32/u8 *before* IR
mode, so the first kernel to adopt IR mode does not need the barrier
mechanism on day one. Barriers stay in the roadmap for later kernels
that need BF16/F16/F8 type conversion the DSL doesn't own.

## Invariants to preserve from day one

These are not extra work. They are the same design, named carefully so
future extensions are additive rather than rewrites. Status of each on the
current branch is noted.

1. **SSA at the IR level.** Every `Op.def` is a fresh value id. The DSL
   rebinds handles on `operator=`, never mutates in place. Every future
   optimization assumes this.
   *Status: holds at record time. `TwoAddressPass` deliberately breaks it
   afterwards (one id, several defs, several `VNInfo`s) — the same
   post-two-address form LLVM uses. Any pass added after `TwoAddress`
   must therefore not assume single-def.*

2. **Op is a plain struct with room to grow.** Adding fields
   (`remat`, `preferred_reg`, `priority_hint`) later doesn't break
   recording sites. Use aggregate initialization, no constructors.

3. **Allocator is a swappable pass** over `(IR, config) → Assignment`.
   Not coupled to IR recording. Swap linear scan for anything else later
   without touching DSL code.

4. **Spill-victim is a swappable function.** Interface:
   `Interval pick_victim(const ActiveSet&, op_index)`. Minimum: furthest
   next use. Later: priority-weighted, remat-aware. Ten-line swap.
   *Status: not honoured. Remat victim selection and Belady spill victim
   selection are both inlined in `assign_registers`' failure path.*

5. **Lowering is mechanical substitution only.** Walk IR, call
   `emit(regs)`. No allocator state reads in this pass.
   *Status: holds, with one deliberate exception — `is_copy` ops are
   emitted by the lowering walk itself (class-dispatched `mov` /
   `uni_vmovups`, skipped entirely when def and source coalesced), which
   is how a coalesced copy costs nothing.*

## Remat expansion plan

The current same-list suffix policy is the correctness baseline. Planned
expansion should proceed in increasing order of semantic difficulty:

1. **Sibling branches via separate per-branch remats**
   - What it means:
     If a value is first needed in multiple sibling branch bodies, insert
     separate remats in each branch rather than trying to share one clone.
   - Why this is the next step:
     It preserves path locality and avoids cross-branch dominance
     reasoning.
   - Main risk:
     code-size growth, not correctness.

2. **Local child-region remat when first used there**
   - What it means:
     If a later use occurs inside a nested child region, remat inside
     that child at the first use there instead of trying to rewrite from
     the parent.
   - Why this is still relatively safe:
     the clone remains local to one region body.
   - Main risk:
     duplicated remats across many child regions.

3. **Careful parent-list hoisting on one straight-line path**
   - What it means:
     If uses occur in a parent list after a nested region, allow remat
     placement in the parent list when the rewritten uses stay on one
     straight-line path dominated by that insertion point.
   - Why it is harder:
     remat placement and rewritten uses are no longer in the same list.
   - Main risk:
     accidentally lengthening the clone so much that the pressure win
     disappears.

4. **Nested cross-region shared remat**
   - What it means:
     One clone serves uses across multiple nested regions / structured
     paths.
   - Why it is much harder:
     now real dominance/path reasoning is needed.
   - Main risk:
     the transformation becomes CFG-sensitive and starts resembling true
     live-range splitting rather than local repair.

5. **Loop wraparound / cross-iteration remat**
   - What it means:
     Use a clone inserted at one loop-body position to serve uses that
     are only "later" in runtime because they occur in the next
     iteration.
   - Why it is the hardest case:
     flat body op indices do not encode iteration number. "Earlier in the
     loop body" and "later in the next iteration" are distinct runtime
     concepts that the current IR does not model.
   - Main risk:
     this is exactly the class of bug that caused the original wrong
     answers. Do not reintroduce it without explicit loop/backedge
     semantics.

In practice, implementation should likely stop at steps 1-3 unless a
measured kernel clearly requires more. Step 5 in particular should be
treated as "requires new IR/analysis machinery", not as a small policy
tweak.

## Extensibility — what lands incrementally

With the invariants above, each of the following is an additive change
that does not require rewriting anything shipped in the minimum:

| Feature | Rough cost | What it touches | Status |
|---|---|---|---|
| Rematerialize constants/broadcasts instead of spilling | ~30 lines | New `remat` field on `Op`; `pick_victim` prefers remat when set | **shipped**, integrated in `assign_registers`; no `remat` field — rematerializability is derived from the def op's reads |
| Priority-weighted spill heuristics (loop depth, use count) | ~10 lines | Swap `pick_victim` | not started (spill itself is a stub) |
| Register preferences (hint allocator to reuse a specific reg) | ~30 lines | New `preferred` field on `Op`; scan honors at assignment | partially — `copy_of` acts as a preference for copies and tied ops |
| Proper copy coalescing beyond trivial | ~100 lines | New pre-pass before linear scan; merges compatible intervals | not started; `TwoAddressPass` + `copy_of` cover the FMA seed case |
| Live range splitting at cold regions | ~100 lines | Extend `Interval` with split points; scan splits in high-pressure windows | not started |
| Escape-hatch barriers for legacy emitters | ~50 lines | New `Op` with `clobbers` mask; spill/reload forced around it | not needed yet (DSL owns f32/u8/f16/bf16) |
| Mask (k-register) allocation | ~50 lines | third pool + `RegisterClass::Mask` | **shipped**; per-target allocation order from `vector_target::predicate_pool()` |
| Scratch-register modelling for emit closures | ~30 lines | extra dead def on the op that needs scratch | **shipped**; scratch is an IR value, and `Op::early_clobber` stops the allocator handing it a live read's register |
| Register tuple constraints (consecutive registers) | ~150 lines | a tuple register class, or a constraint on a value group; assignment must place members adjacently | not started, and required by SVE `ST3W {z0-z2}` and RVV segment ops. LLVM's answer is tuple register classes (`ZPR3`, `ZPR4`). Until then `supports_masked_interleaved_access()` is false on every target and interleaved stores keep the stack-slot fallback |

## What would force a rewrite (probably never needed)

- **Graph coloring / PBQP allocator.** Different data structures,
  different algorithm. Linear scan is ~10% off optimal at our scale, and
  the gap vanishes with 16+ registers and no cross-function pressure.
  Do not plan for this.
- **Non-SSA IR.** Would require dominance info and mutable-value
  tracking. The by-value DSL rule prevents the need.
- **Unstructured control flow.** We only have `_if` and `foreach`, both
  structured. Arbitrary jumps would force a real CFG with block nodes.
- **Reordering transformations** (LICM, unrolling, fusion). The region
  representation is sufficient for allocation but not for reordering.
  Loop unrolling was nonetheless added on top of it, and it shows: the
  cloner cannot clone nested regions, and the pass has to find the loop
  footer by matching the debug string `"loop_footer"`. Both are symptoms
  of the missing CFG, not of unrolling being hard. The fix is a real CFG
  (~100 lines built from the region tree), not a rewrite of anything
  else.

## Debuggability

Eager mode debugging: dump disassembly, read top to bottom, match source
lines. IR mode loses this — the xbyak emitted after allocation may
reference different registers than any locally visible DSL line would
suggest. The debug story has to be the IR dump itself.

**As implemented:** `dump_ops` + `dump_assignment`, driven by
`OV_JIT_IR_DUMP`, plus `OV_JIT_IR_TRACE` / `OV_JIT_IR_TRACE_REMAT` traces
of range construction, allocation and lowering. Output goes to `std::cout`
unconditionally when the variable is set — the dumps are **not** guarded by
`ENABLE_DEBUG_CAPS`, and there is no pressure-curve dump. Live ranges are
printed with segments and assigned physical registers, which is the one
debug facility that has actually paid for itself.

**Original sketch of the minimum tooling** (~100 lines):

- `ir.dump()` — prints the recorded op stream with value ids:
  ```
  %0  def  load(%ptr_y)                [y_src]
  %1  def  load(%ptr_u)                [u_src]
  %2  def  broadcast(%const_idx=4)     [c_yr]
  %3  def  fma(%0, %2, %bias)          [y_scaled]
  ...
  ```
- `ir.dump_intervals()` — after allocation, prints value-id to physical
  register mapping and interval bounds. Spills and reloads show up as
  inserted ops.
- `ir.dump_pressure_curve()` — prints per-op-index live count for each
  pool. Peak values indicate where the kernel is tightest. Regression
  detection: if a refactor bumps peak from 11 to 14, this shows it.

The dumps are guarded under `ENABLE_DEBUG_CAPS`. Production builds pay
zero runtime cost.

## What IR mode is explicitly not

Restated from `jit_kernel.md` to prevent scope creep in this doc, with the
two entries that reality already overtook marked as such:

- **Not a compiler.** No CSE, no LICM, no constant folding, no dead-code
  elimination, no pattern matching, no instruction selection beyond what
  the DSL primitives already committed to. *Still true for those
  transforms; but the pipeline does contain unrolling, remat, two-address
  lowering and (via record-time mode flags) tail synthesis, so "not a
  compiler" now means "no value-level optimization", not "nothing but
  allocation".*
- ~~**Not a default.** Kernels stay on eager mode unless measurement shows
  they benefit from IR mode.~~ *Superseded: eager vector mode is gone. Any
  kernel using vector `variable`s uses IR mode.*
- **Not a replacement for `jit_load_emitter` and friends.** Legacy
  emitters continue to handle BF16, F16, F8/F4, and other exotic type
  pairs. IR mode accesses them via the barrier mechanism or, preferably,
  via DSL-native load/store for the types the DSL owns.
- **Not a path to std::datapar code quality parity.** That requires the
  full C++ compiler middle-end (inlining, autovectorization refinement,
  scalar CSE). IR mode matches register allocation quality only. Real
  parity requires MLIR or LLVM integration, which is the strategic
  decision declined in `jit_kernel.md`, "Why not MLIR" section.
- **Not something every kernel needs.** The whole premise is that
  register allocation is the one thing a kernel author cannot easily
  manage manually. Kernels that don't have a pressure problem don't need
  the tool.

## Implementation ordering — outcome

Original plan and what happened:

1. **Cheap tier debug counters** under `ENABLE_DEBUG_CAPS` — *skipped.*
   The IR dump replaced it; `peak_live` is computed in `Assignment` but
   nothing consumes it.
2. **Pool cap fix** (vector pool separate from the GPR index range,
   `zmm16..zmm31` on AVX-512) — *not done.* `jit_kernel.cpp:346-354` still
   derives the vector pool from `RAX..R15` and `zmmregs()` still stops at
   `zmm15`. Still the cheapest available win.
3. **DSL-native `load`/`store`** — *shipped*, and extended past the plan
   to f16/bf16.
4. **IR mode — record + allocate + lower** — *shipped*, plus GPR
   allocation, remat, two-address lowering, unrolling and epilogue
   synthesis that were not in the plan.
5. **Barrier mechanism** — *not needed* so far.
6. **Extensions** — see the status column in the table above.

Current queue is in `jit_kernel_journal.md`, "Immediate work queue": make
spill safe, add a verifier and randomized/differential tests, replace the
liveness heuristics with real dataflow, then remove the record-time mode
flags in favour of an epilogue pass.

## Open questions — status

- **Spill slot sharing.** Still open, and premature: spill itself is a
  stub. `PassContext::spill_slots` allocates one slot per spill with no
  reuse; slot offsets are never even computed.
- **Vector pool vs mask pool on AVX-512.** Resolved: masks are a register
  class with their own allocation order, so several constructs can consume
  predicates in one kernel. Spilling a predicate is still unaddressed —
  there is no spiller for any class, and a k-register spill needs a GPR
  detour.
- **Value-id reuse on operator=.** Settled: ids are never reused; the
  value-id space grows monotonically, and remat/two-address/unroll all
  allocate fresh ids. Fine at kernel scale.
- **IR mode entry/exit syntax.** Settled: explicit `begin_ir()` /
  `end_ir()` on the kernel, no RAII guard, no env gating. `ir_mode()`
  reports whether `_ir` is live. Since eager vector mode is gone, the
  brackets are effectively mandatory for any kernel doing vector work,
  which argues for folding them into `preamble()`/`postamble()` later.
- **GPR pool pressure.** New question, not in the original list. With
  GPRs allocator-managed and the pool at ~10 allocable registers,
  GPR-heavy expansions (`ir_load_partial`, `ir_memcpy`) can exhaust it —
  and with spill unimplemented, exhaustion is a hard failure. Either
  make spill real or reduce those expansions (masked partial access
  instead of stack bounce).

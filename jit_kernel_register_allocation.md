# Register allocation for `jit_kernel`

Design notes for the lightweight register allocator that will run as an
opt-in IR mode on top of `src/plugins/intel_cpu/src/nodes/kernels/x64/jit_kernel.hpp`.
Companion to `jit_kernel.md` — specifically the "Register allocation and
pressure analysis" section, which motivates why this is needed.

Scope is deliberately narrow: **this is a register allocator, not a
compiler.** No CSE, no LICM, no constant folding, no instruction selection,
no pattern matching. One DSL call = one emitted instruction, same as
today's eager mode. The only thing IR mode changes is *which physical
registers* the instructions reference.

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

Both gaps are things a classical linear-scan allocator handles naturally.
Both gaps are things the author cannot easily manage by hand — reasoning
about global liveness and split points while writing locally readable
code is the one thing the DSL cannot leave to the author.

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

## Two modes, one DSL

- **Eager mode** (today's default). Every DSL call emits xbyak
  immediately. Scope-based `shared_reg` pool handles lifetimes. Fast to
  debug (disassembly matches source line-by-line). Composes trivially
  with legacy emitters (`jit_load_emitter` etc.). Used by the majority
  of kernels.
- **IR mode** (opt-in, narrow). DSL calls record into a per-kernel IR
  without emitting. A second pass runs linear-scan allocation and then
  emits xbyak with the allocated physical registers. Used by kernels
  fighting register pressure — color_convert is the motivating example.

Mode is picked per kernel (or per region within a kernel). There is no
"upgrade every kernel to IR mode" roadmap. Eager mode is the correct
default for loop-heavy kernels with simple dataflow; the abstraction cost
of IR mode (debug via IR dump, not xbyak at emit time) is only worth
paying when measurement shows pressure symptoms.

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

One `op()` function per arity. Handles mode switch, IR recording, eager
execution, and register width — **written once, covers all instructions
of that arity**:

```cpp
variable op(Insn2 insn, const variable& a, const variable& b) {
    if (ir_mode()) {
        return variable(ir_.def({a.id(), b.id()},
            [this, insn](const EmitContext& ctx) {
                auto [d, r] = ctx.regs<2>();
                lower(insn, d, r[0], r[1]);
            }));
    }
    variable dst = alloc_reg();
    lower(insn, dst.reg(), a.reg(), b.reg());
    return dst;
}
```

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

- **Eager mode:** one branch (`ir_mode()`) + one switch (`lower()`)
  per instruction. The switch optimizes to a jump table. Negligible
  against xbyak's own encoding cost.
- **IR mode:** ~2× JIT compilation time (record + emit). Zero effect
  on generated kernel execution — identical machine code either way.
- **C++ compilation:** no templates, no macros, no heavy headers.
  Comparable to existing `uni_*` method definitions.

## IR shape

Minimal. Three data structures.

```cpp
using value_id = uint32_t;

struct Op {
    small_vec<value_id, 3> reads;
    std::optional<value_id> def;
    std::function<void(const EmitContext&)> emit;  // captures insn + lowering call
    uint32_t clobbers = 0;   // bitmask, for escape-hatch barriers
    bool is_copy = false;    // for trivial coalescing
};

struct CFGMarker {
    enum Kind { Label, Branch, Jmp, LoopBegin, LoopEnd };
    Kind kind;
    label_id target;
};

struct IR {
    std::vector<std::variant<Op, CFGMarker>> stream;
};
```

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

### Pass 1: Liveness dataflow

Backward pass over basic blocks computing per-block `live_in` / `live_out`
sets. Fixpoint over `foreach` backedges — iterate until no set changes.

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

### Predicated loops (`foreach_predicated`)

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

## The one ugly corner: escape-hatch barriers

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
future extensions are additive rather than rewrites.

1. **SSA at the IR level.** Every `Op.def` is a fresh value id. The DSL
   rebinds handles on `operator=`, never mutates in place. Every future
   optimization assumes this.

2. **Op is a plain struct with room to grow.** Adding fields
   (`remat`, `preferred_reg`, `priority_hint`) later doesn't break
   recording sites. Use aggregate initialization, no constructors.

3. **Allocator is a swappable pass** over `(IR, config) → Assignment`.
   Not coupled to IR recording. Swap linear scan for anything else later
   without touching DSL code.

4. **Spill-victim is a swappable function.** Interface:
   `Interval pick_victim(const ActiveSet&, op_index)`. Minimum: furthest
   next use. Later: priority-weighted, remat-aware. Ten-line swap.

5. **Lowering is mechanical substitution only.** Walk IR, call
   `emit(regs)`. No allocator state reads in this pass.

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

| Feature | Rough cost | What it touches |
|---|---|---|
| Rematerialize constants/broadcasts instead of spilling | ~30 lines | New `remat` field on `Op`; `pick_victim` prefers remat when set |
| Priority-weighted spill heuristics (loop depth, use count) | ~10 lines | Swap `pick_victim` |
| Register preferences (hint allocator to reuse a specific reg) | ~30 lines | New `preferred` field on `Op`; scan honors at assignment |
| Proper copy coalescing beyond trivial | ~100 lines | New pre-pass before linear scan; merges compatible intervals |
| Live range splitting at cold regions | ~100 lines | Extend `Interval` with split points; scan splits in high-pressure windows |
| Escape-hatch barriers for legacy emitters | ~50 lines | New `Op` with `clobbers` mask; spill/reload forced around it |

## What would force a rewrite (probably never needed)

- **Graph coloring / PBQP allocator.** Different data structures,
  different algorithm. Linear scan is ~10% off optimal at our scale, and
  the gap vanishes with 16+ registers and no cross-function pressure.
  Do not plan for this.
- **Non-SSA IR.** Would require dominance info and mutable-value
  tracking. The by-value DSL rule prevents the need.
- **Unstructured control flow.** We only have `_if` and `foreach`, both
  structured. Arbitrary jumps would force a real CFG with block nodes.
- **Reordering transformations** (LICM, unrolling, fusion). The CFG
  markers-in-stream representation is sufficient for allocation but not
  for reordering. If reordering ever becomes interesting, the fix is a
  real CFG (~100 lines to reconstruct from the marker stream), not a
  rewrite of anything else.

## Debuggability

Eager mode debugging: dump disassembly, read top to bottom, match source
lines. IR mode loses this — the xbyak emitted after allocation may
reference different registers than any locally visible DSL line would
suggest. The debug story has to be the IR dump itself.

**Minimum debug tooling** (~100 lines):

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

Restated from `jit_kernel.md` to prevent scope creep in this doc:

- **Not a compiler.** No CSE, no LICM, no constant folding, no dead-code
  elimination, no pattern matching, no instruction selection beyond what
  the DSL primitives already committed to.
- **Not a default.** Kernels stay on eager mode unless measurement shows
  they benefit from IR mode.
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

## Implementation ordering

1. **Cheap tier debug counters** (from `jit_kernel.md` roadmap item 2).
   ~50 lines. Peak tracker + allocation counter under `ENABLE_DEBUG_CAPS`.
   Ships first — baseline measurement for every kernel under
   consideration, regression detection once IR mode lands.
2. **Pool cap fix** (from `jit_kernel.md` roadmap item 1). Separate the
   vector pool size from the GPR loop limit; populate 32 entries on
   AVX-512. ~10-20 lines. Relaxes pressure on its own without any new
   infrastructure.
3. **DSL-native `load` / `store` for f32 and u8**. ~250 lines. Prereq
   for IR mode adoption on color_convert without the barrier mechanism.
4. **IR mode — record + linear scan + lower** (this document's core).
   ~600 lines including debug dumps. Ship with IR mode enabled on
   color_convert. Validate: pressure curve dump shows the per-iteration
   broadcasts and permute loads hoisted to the preamble; peak live count
   drops to theoretical floor.
5. **Barrier mechanism** (~50 lines). Lands when a second kernel adopts
   IR mode and needs a type conversion the DSL doesn't own.
6. **Extensions from the table above.** Each driven by a concrete kernel
   that benefits, not speculatively.

Steps 1 and 2 ship independently and benefit every kernel. Step 3 is
independently useful as a load/store upgrade even without IR mode. Step
4 is the committed investment. Steps 5-6 are measurement-gated.

## Open questions — revisit at implementation time

- **Spill slot sharing.** Can two non-overlapping spilled intervals
  share a stack slot? Yes, trivially (they're non-overlapping). Worth
  it in the minimum? Only if total spill volume is material. Measure
  first.
- **Vector pool vs mask pool on AVX-512.** Mask k-regs (k0..k7) are
  scarce and the comparison-heavy kernels would exhaust them first.
  Does the mask pool need its own `pick_victim` tuning? Probably — a
  k-reg spill is more expensive (GPR detour) than a vector spill.
  Revisit when the first mask-heavy kernel adopts IR mode.
- **Value-id reuse on operator=.** Strict SSA says every rebind is a
  new id even if the old id is dead. Pragmatic: if the old value's
  interval ended, its id can be reused directly by the new def. The
  allocator produces the same result either way; the difference is the
  size of the value-id space. Probably not worth the complexity to
  reuse.
- **IR mode entry/exit syntax.** `k.with_ir_mode([&]{ ... })` vs a flag
  on the kernel constructor vs a per-region RAII guard. Bikeshed at
  implementation time.

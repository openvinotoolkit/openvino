// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// jit_kernel IR and register allocator — see jit_kernel_register_allocation.md
// for the design and jit_kernel_journal.md for current status and hazards.
//
// Shape: op list with nested regions (MLIR-style) for control flow, LLVM-style
// live ranges (segments + value numbers, half-open, early/late slots), two
// register classes (Vec, GPR) over two physical pools, interference-based
// assignment with coalescing and integrated rematerialization, and a pass
// pipeline (two-address -> liveness -> allocate -> verify -> lower).
//
// Invariants:
//   - Mostly SSA at record time: Op.def is a fresh value id except where
//     def_into() deliberately redefines an existing one, which is how a
//     loop-carried accumulator is expressed. TwoAddressPass breaks
//     single-def further (one id, several defs) exactly as LLVM does
//     post-two-address; no pass may assume single-def.
//   - Op is a plain struct with room to grow.
//   - Allocation is a pass over (IR, PassContext) -> Assignment.
//   - Lowering is mechanical substitution: walk the IR, resolve reads/def to
//     physical registers, call the op's emit closure. The one exception is
//     is_copy ops, which lowering emits itself (and elides when coalesced).
//   - No spiller: pool exhaustion with nothing rematerializable throws.

#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace ov::intel_cpu::jit_kernel_ir {

using value_id = std::uint32_t;
inline constexpr value_id invalid_value = std::numeric_limits<value_id>::max();

// LLVM-style register class. The allocator handles all classes in one
// unified work list, picking from the physical pool of the value's class.
// Class is stored per virtual register (Op::def_rc / LiveRange::rc), NOT
// on PhysReg — matching LLVM's MachineRegisterInfo design.
//
// Mask is a real class, not a special case: predicate registers are as
// allocatable as any other file, and every target constrains them
// differently (x86 cannot use k0 as a write-mask — LLVM spells that
// VK*WM; SVE restricts governing predicates to p0-p7 — PPR_3b; RVV
// requires masks in v0 — a one-register class, VMV0). Those constraints
// live in the allocation order, so the allocator needs no target
// knowledge.
enum class RegisterClass : std::uint8_t {
    Vec,    // SIMD register file (xmm/ymm/zmm, v, z)
    GPR,    // general-purpose register file
    Mask,   // predicate register file (k, p, or v0 on RVV)
};

inline constexpr std::size_t register_class_count = 3;

inline const char* to_string(RegisterClass rc) noexcept {
    switch (rc) {
    case RegisterClass::GPR:  return "gpr";
    case RegisterClass::Mask: return "mask";
    default:                  return "vec";
    }
}

// A lightweight physical register handle — just an index into the hardware
// register file. The register class is determined by the virtual register
// (value_id), not by the PhysReg itself.
struct PhysReg {
    std::uint32_t idx = 0;

    friend bool operator==(PhysReg a, PhysReg b) noexcept { return a.idx == b.idx; }
    friend bool operator!=(PhysReg a, PhysReg b) noexcept { return a.idx != b.idx; }
};

// Arguments threaded into an emit closure at lowering time.
// In Slice 1 no closure is actually invoked end-to-end; the struct exists so
// the closure signature is stable and Slice 2 can plug into it.
//
// `reads` is a plain vector rather than std::span because the intel_cpu
// plugin is C++17 and std::span is C++20. Switch to std::span when the
// plugin moves to C++20.
// A memory operand folded into an instruction: base register plus a byte
// displacement. Arch-neutral on purpose — the emit closure turns it into
// whatever address form the target uses.
struct FoldedMem {
    PhysReg base;
    std::uint32_t offset = 0;
    int read = -1;        // which read of the op became the memory operand
};

struct EmitContext {
    std::optional<PhysReg> def;              // physical reg backing Op.def, if any
    const std::vector<PhysReg>& reads;       // physical regs backing Op.reads, in order
    std::optional<FoldedMem> folded;         // set when an operand was folded into memory
};

using EmitFn = std::function<void(const EmitContext&)>;

class IR;  // forward declaration for Op::body

// The one and only op type. The allocator reads only the dependency shape
// (reads, def, tied_to, def_rc); everything else is for lowering or debug.
struct Op {
    std::vector<value_id> reads;
    value_id def = invalid_value;        // invalid_value = no def (e.g. store)
    EmitFn emit;                         // called by lowering with allocated regs
    bool is_copy = false;                // trivial coalescing hint for pure copies
    int tied_to = -1;                    // LLVM-style tied operand: index into reads[]
                                         // that def must share a register with. -1 = none.
                                         // The allocator coalesces or inserts a copy.
    RegisterClass def_rc = RegisterClass::Vec;  // register class for the def value

    // ── Memory effects ────────────────────────────────────────────────
    // LLVM spells these mayLoad / mayStore on MachineInstr, generated from
    // the target description. They exist here for the same reason: no pass
    // may reorder a memory access past another one without them.
    bool may_load = false;
    bool may_store = false;
    int mem_ptr_read = -1;               // index into reads[] holding the base pointer
    std::uint32_t mem_offset = 0;        // byte displacement of the access

    // ── Foldable operands ─────────────────────────────────────────────
    // `foldable_reads` is a bitmask of the operands this op can take from
    // memory instead of a register, and `fold_emit` is the memory form of
    // the instruction. That is the rr/rm instruction pair an x86 target
    // declares in its .td file, expressed as two closures because the DSL
    // — not the IR — knows the instruction.
    //
    // More than one bit may be set for a commutative op: x86 requires the
    // memory operand to come last, so folding a first operand means
    // swapping the sources. LLVM does the same thing with isCommutable +
    // commuteInstruction; here the fold closure reads
    // EmitContext::folded->read and picks the surviving register operand.
    std::uint8_t foldable_reads = 0;
    EmitFn fold_emit;

    // Set by FoldMemoryOperandsPass: the read index whose value now comes
    // from memory. reads[folded_read] holds the base pointer.
    int folded_read = -1;

    bool early_clobber = false;          // def is written before the reads are
                                         // consumed, so it must not share a
                                         // register with any of them. Mirrors
                                         // LLVM's early-clobber operand: the
                                         // def starts at the early slot, which
                                         // makes it interfere with its own reads.
    std::unique_ptr<IR> body;            // non-null = region op (loop, branch)
    bool is_loop = false;                // true = extend intervals across body (repeats)
    const char* name = "";               // debug tag for dump (not used by allocator)
};

// Builder / container for the recorded op stream.
// Ops record into the current insertion target (top-level by default).
// Loop ops contain a nested IR (body) — like MLIR's regions.
class IR {
public:
    // Record an op that defines a fresh value. Returns the new value id.
    // RegisterClass specifies which register file (Vec or GPR).
    value_id def(std::vector<value_id> reads, EmitFn emit, const char* name = "",
                 RegisterClass rc = RegisterClass::Vec) {
        const value_id id = _next_value++;
        Op op;
        op.reads = std::move(reads);
        op.def = id;
        op.emit = std::move(emit);
        op.def_rc = rc;
        op.name = name;
        target().push_back(std::move(op));
        return id;
    }

    // Record an op whose def is written before its reads are consumed —
    // a multi-instruction expansion that scribbles on the destination
    // first. The def gets the early slot, so it interferes with the reads
    // and the allocator gives it a different register.
    value_id def_early_clobber(std::vector<value_id> reads, EmitFn emit,
                               const char* name = "",
                               RegisterClass rc = RegisterClass::Vec) {
        const value_id id = _next_value++;
        Op op;
        op.reads = std::move(reads);
        op.def = id;
        op.emit = std::move(emit);
        op.def_rc = rc;
        op.early_clobber = true;
        op.name = name;
        target().push_back(std::move(op));
        return id;
    }

    // Record an op with a tied-operand constraint: def must share a
    // register with reads[tied_to]. The allocator coalesces if possible,
    // otherwise the emit closure must handle the mismatch (e.g. vmovups).
    value_id def_tied(std::vector<value_id> reads, int tied_to, EmitFn emit, const char* name = "",
                      RegisterClass rc = RegisterClass::Vec) {
        const value_id id = _next_value++;
        Op op;
        op.reads = std::move(reads);
        op.def = id;
        op.emit = std::move(emit);
        op.tied_to = tied_to;
        op.def_rc = rc;
        op.name = name;
        target().push_back(std::move(op));
        return id;
    }

    // Record an op that redefines an existing value instead of creating a
    // new one: one value id, several defs, the op reading and writing the
    // same register.
    //
    // This is what a loop-carried accumulator needs — a value initialized
    // before a loop and updated in place inside it, so the update survives
    // the back edge. Recording a fresh def instead (the ordinary
    // `def_tied` path) produces a body that reads the initial value every
    // trip and throws its own result away, which compiles, allocates and
    // runs while computing the wrong thing.
    //
    // LLVM spells this as a PHI at the loop header joined by the preheader
    // and the latch; PHIElimination and TwoAddressInstructionPass then
    // lower it to exactly this form. TwoAddressPass already produces the
    // same shape here for tied operands — the difference is only that the
    // initializing def belongs outside the loop, which is the caller's
    // job and cannot be expressed by a pass that only sees the tie.
    //
    // No tied_to: the constraint is satisfied by construction, since the
    // op reads the value it defines.
    void def_into(value_id existing, std::vector<value_id> reads, EmitFn emit,
                  const char* name = "", RegisterClass rc = RegisterClass::Vec) {
        OPENVINO_ASSERT(existing != invalid_value && existing < _next_value,
                        "def_into: not an existing value");
        // An in-place update has to read what it updates. A value that
        // does not depend on its previous contents wants a fresh def, and
        // writing it through def_into would hide a dead initialization
        // from the reader as much as from the verifier — which has no
        // dominance check to catch it.
        OPENVINO_ASSERT(std::find(reads.begin(), reads.end(), existing) != reads.end(),
                        "def_into: the updated value must appear among the reads");
        Op op;
        op.reads = std::move(reads);
        op.def = existing;
        op.emit = std::move(emit);
        op.def_rc = rc;
        op.name = name;
        target().push_back(std::move(op));
    }

    // Record an op that reads values but defines none (e.g. a store).
    void use(std::vector<value_id> reads, EmitFn emit, const char* name = "") {
        Op op;
        op.reads = std::move(reads);
        op.emit = std::move(emit);
        op.name = name;
        target().push_back(std::move(op));
    }

    // Record a copy-like op: defines a fresh value whose contents come from a
    // single source. Flagged for the allocator's trivial coalescing pass.
    value_id copy(value_id src, EmitFn emit, const char* name = "",
                  RegisterClass rc = RegisterClass::Vec) {
        const value_id id = _next_value++;
        Op op;
        op.reads = {src};
        op.def = id;
        op.emit = std::move(emit);
        op.is_copy = true;
        op.def_rc = rc;
        op.name = name;
        target().push_back(std::move(op));
        return id;
    }

    // Record a region op — a nested body of ops. The emit closure is
    // called at lowering time before the body ops are lowered.
    // `is_loop` controls whether intervals are extended across the body.
    // `reads` are values consumed by the region header (e.g. loop counter
    // and end value for the cmp/jge). Resolved at lowering time via
    // EmitContext::reads, same as regular ops. LLVM-style: the loop
    // header is part of the region op, not a separate instruction.
    template <typename BodyBuilder>
    void region(std::vector<value_id> reads, EmitFn emit, BodyBuilder&& body_builder, bool is_loop = false) {
        auto body = std::make_unique<IR>();
        // Save/restore cursor so nested regions (e.g. ir_if inside a loop)
        // don't clobber the outer cursor.
        auto* saved_cursor = _cursor;
        _cursor = &body->_ops;
        _loop_depth += static_cast<unsigned>(is_loop);
        body_builder();
        _loop_depth -= static_cast<unsigned>(is_loop);
        _cursor = saved_cursor;

        Op op;
        op.reads = std::move(reads);
        op.emit = std::move(emit);
        op.body = std::move(body);
        op.is_loop = is_loop;
        target().push_back(std::move(op));
    }

    // No-reads overload for backward compatibility (ir_if, branches).
    template <typename BodyBuilder>
    void region(EmitFn emit, BodyBuilder&& body_builder, bool is_loop = false) {
        region({}, std::move(emit), std::forward<BodyBuilder>(body_builder), is_loop);
    }

    // Record a loop region with reads (e.g. loop counter + end value).
    template <typename BodyBuilder>
    void loop(std::vector<value_id> reads, EmitFn emit, BodyBuilder&& body_builder) {
        region(std::move(reads), std::move(emit), std::forward<BodyBuilder>(body_builder), /*is_loop=*/true);
    }

    // No-reads loop overload for backward compatibility.
    template <typename BodyBuilder>
    void loop(EmitFn emit, BodyBuilder&& body_builder) {
        region({}, std::move(emit), std::forward<BodyBuilder>(body_builder), /*is_loop=*/true);
    }

    // Is the recording cursor inside a loop body? A body recorded once and
    // executed many times cannot have induction updates folded into
    // constants, so anything that wants to constant-fold an update has to
    // ask first. LLVM answers the same question with LoopInfo; here the
    // cursor already knows, because control flow is structural.
    [[nodiscard]] bool recording_in_loop() const { return _loop_depth > 0; }

    // The op just recorded. Builder-style annotation for the properties a
    // recording site knows but the generic builders do not take — memory
    // effects and the foldable-operand pair. Mirrors MachineInstrBuilder
    // chaining onto the instruction it just built.
    Op& last() { return target().back(); }

    [[nodiscard]] const std::list<Op>& ops() const noexcept { return _ops; }
    [[nodiscard]] std::list<Op>& ops() noexcept { return _ops; }
    [[nodiscard]] std::size_t value_count() const noexcept { return _next_value; }
    void set_value_count(value_id v) noexcept { _next_value = v; }

    void dump(std::ostream& os) const;

private:
    std::list<Op>& target() { return _cursor ? *_cursor : _ops; }

    std::list<Op> _ops;
    std::list<Op>* _cursor = nullptr;    // non-null = recording into a body
    unsigned _loop_depth = 0;            // enclosing loop regions at the cursor
    value_id _next_value = 0;
};

// Value number info — tracks one definition point within a LiveRange.
// Post-SSA (after TwoAddressPass), a single value_id may have multiple
// defs (COPY + FMA). Each def gets its own VNInfo.
// Mirrors llvm::VNInfo.
struct VNInfo {
    std::uint32_t id = 0;    // unique within the LiveRange
    std::uint32_t def = 0;   // slot index of the defining op
};

// A half-open [start, end) range within a LiveRange. LLVM naming.
// Each segment is tagged with the VNInfo that produced its value.
// Mirrors llvm::LiveRange::Segment.
struct Segment {
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    std::uint32_t valno = 0;  // index into LiveRange::valnos

    [[nodiscard]] bool contains(std::uint32_t index) const noexcept {
        return start <= index && index < end;
    }

    [[nodiscard]] bool overlaps(std::uint32_t s, std::uint32_t e) const noexcept {
        return start < e && s < end;
    }

    [[nodiscard]] bool overlaps(const Segment& other) const noexcept {
        return start < other.end && other.start < end;
    }
};

// Live range for a single value id — a sorted list of non-overlapping
// segments, each tagged with a VNInfo. Mirrors llvm::LiveRange.
//
// In SSA form: one VNInfo, one or more segments (branches produce
// multiple segments for one def). Post-SSA (after TwoAddressPass):
// multiple VNInfos, one per def of the same value_id.
struct LiveRange {
    value_id id = invalid_value;
    std::vector<Segment> segments;   // sorted by start, non-overlapping
    std::vector<VNInfo> valnos;      // value numbers, one per def
    value_id copy_of = invalid_value;  // coalescing hint
    RegisterClass rc = RegisterClass::Vec;  // register class (from defining Op)

    // First segment's start. Returns max uint32 if empty.
    [[nodiscard]] std::uint32_t beginIndex() const noexcept;
    // Last segment's end (half-open). Returns 0 if empty.
    [[nodiscard]] std::uint32_t endIndex() const noexcept;
    // True if `index` falls within any segment. Mirrors LLVM LiveRange::liveAt.
    [[nodiscard]] bool liveAt(std::uint32_t index) const noexcept;
    // True if any segment overlaps with any segment of `other`.
    // Mirrors LLVM LiveRange::overlaps(const LiveRange&).
    [[nodiscard]] bool overlaps(const LiveRange& other) const noexcept;
    // True if any segment overlaps [start, end).
    [[nodiscard]] bool overlaps(std::uint32_t start, std::uint32_t end) const noexcept;
    // Allocate a new VNInfo for a def at the given slot. Returns its index.
    std::uint32_t getNextValue(std::uint32_t def);
    // Insert a segment, maintaining sorted order. Merges overlapping
    // or adjacent segments (half-open: [0,2) + [2,4) → [0,4)).
    void addSegment(Segment s);
    // True if no segments exist.
    [[nodiscard]] bool empty() const noexcept;
};

// Live range construction. Walks the op tree recursively — loop bodies are
// traversed inline and ranges of values live inside a loop are extended
// to cover the full loop body. Sibling branch bodies produce separate
// segments for values used in different branches.
std::vector<LiveRange> compute_live_ranges(const IR& ir);

// Result of linear scan.
struct Assignment {
    std::unordered_map<value_id, PhysReg> reg;
    std::uint32_t peak_live = 0;         // max simultaneously-allocated regs
};

// Raised when the pool overflows and no value is rematerializable. There is
// no spiller, so this is a hard failure: the kernel author must reduce
// pressure (fewer live values, smaller unroll, restructured body).
class allocation_failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Interference-based register assignment with integrated rematerialization.
//
// Not linear scan (no active list, no expiry): live ranges are visited in
// order of their first slot and each takes the first physical register in
// its class whose assigned segments do not overlap. This is the
// interference test LLVM's allocators use, with first-fit selection
// instead of a priority queue.
struct PassContext;  // forward declaration

// Returns an Assignment on success, or std::nullopt when the IR was
// rewritten (a value was rematerialized) — the caller must recompute live
// ranges and retry. Throws allocation_failure when a register is needed
// and nothing can be rematerialized; there is no spiller.
std::optional<Assignment> assign_registers(IR& ir,
                                           std::vector<LiveRange>& ranges,
                                           PassContext& ctx);

// Loop unrolling deliberately has no IR pass. The trip count lives inside
// the loop header's emit closure, so a pass over the IR cannot scale it to
// match a cloned body — an earlier attempt silently made kernels process
// factor x the data (heap corruption on RoPE, wrong pixels on
// color_convert). Unrolling therefore stays at recording time, where the
// DSL still knows the bound (foreach_predicated's `unroll` argument). An
// IR-level pass becomes possible once the loop bound and step are IR
// operands of a real loop op instead of captured immediates.

// Test-only: remat a specific value — insert a clone (with reads
// preserved) before each use and rewrite reads.
// Returns true if the IR was modified.
bool unit_test_api_remat_value(IR& ir, value_id vid);

// Verifier — the MachineVerifier equivalent. Checks the invariants that
// allocation and lowering rely on, and throws `verification_failure`
// naming the offending value/op when one is violated. Cheap at kernel
// scale (tens of ops), so it runs unconditionally in the pipeline.
//
// Checked:
//   1. live range structure: segments non-empty, sorted, non-overlapping;
//      register class agrees with the defining op.
//   2. no undefined reads: every read has a defining op in the tree.
//   3. completeness: every value the lowering pass will look up has an
//      assignment (otherwise lowering throws std::out_of_range with no
//      context).
//   4. interference: two values sharing a physical register must have
//      disjoint live ranges. This is the property the allocator exists to
//      guarantee.
//   5. post-pass shape: no tied operand survives TwoAddressPass; copies
//      read exactly one value; assigned registers lie inside their pool.
class verification_failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void verify(const IR& ir,
            const std::vector<LiveRange>& ranges,
            const Assignment& assignment,
            const PassContext& ctx);

// Debug helper: text dump of the op stream. Used by IR::dump and by tests
// to eyeball what was recorded.
void dump_ops(std::ostream& os, const IR& ir);

// Debug helper: dump live ranges and their assigned physical registers.
void dump_assignment(std::ostream& os,
                     const std::vector<LiveRange>& ranges,
                     const Assignment& assignment);

// ── Pass infrastructure ─────────────────────────────────────────────
// LLVM-style pass pipeline. Each pass transforms or analyzes the IR.
// PassContext carries shared state (analysis results, config) between passes.

struct PassContext {
    // Allocation order per register class: the physical registers the class
    // may use, in preference order. Both classes use an explicit list
    // because neither is guaranteed contiguous — GPR excludes rsp/rbp/abi
    // params, and a kernel may have reserved vector registers eagerly
    // before entering IR mode. Mirrors LLVM's AllocationOrder
    // (RegisterClassInfo::getOrder).
    std::vector<std::uint32_t> vec_pool_indices;
    std::vector<std::uint32_t> gpr_pool_indices;
    std::vector<std::uint32_t> mask_pool_indices;   // empty when the ISA has no predicates

    [[nodiscard]] const std::vector<std::uint32_t>& pool(RegisterClass rc) const noexcept {
        switch (rc) {
        case RegisterClass::GPR:  return gpr_pool_indices;
        case RegisterClass::Mask: return mask_pool_indices;
        default:                  return vec_pool_indices;
        }
    }

    // No spill slots: there is no spiller. When the allocator runs out of
    // registers and nothing is rematerializable it throws
    // allocation_failure. Adding a spiller requires frame setup to move
    // behind allocation plus target hooks for store/load of a physical
    // register — see jit_kernel_journal.md, "Immediate work queue".

    // Analysis results — populated by analysis passes, consumed by later passes.
    std::vector<LiveRange> ranges;
    std::optional<Assignment> assignment;

    // Lowering callback — set by jit_kernel before running the pipeline.
    // Called by the lowering pass to emit xbyak instructions.
    using LowerFn = std::function<void(const IR&, const Assignment&)>;
    LowerFn lower_fn;

    // Config
    bool dump = false;     // OV_JIT_IR_DUMP
    bool trace = false;    // OV_JIT_IR_TRACE
    bool disable_memory_folding = false;   // OV_JIT_IR_NO_FOLD, for A/B measurement
};

// Base class for all IR passes.
struct IRPass {
    virtual ~IRPass() = default;
    // Returns true if the IR was modified.
    virtual bool run(IR& ir, PassContext& ctx) = 0;
    virtual const char* name() const = 0;
};

// Pass manager — runs passes in order.
class PassManager {
public:
    template <typename PassT, typename... Args>
    void add(Args&&... args) {
        _passes.push_back(std::make_unique<PassT>(std::forward<Args>(args)...));
    }

    void run(IR& ir, PassContext& ctx) {
        for (auto& pass : _passes) {
            pass->run(ir, ctx);
        }
    }

private:
    std::vector<std::unique_ptr<IRPass>> _passes;
};

// ── Built-in passes ─────────────────────────────────────────────────

// Analysis: compute live ranges from the IR.
struct LiveRangeAnalysis : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "LiveRangeAnalysis"; }
};

// Transform: fold single-use loads into the operand of their consumer.
//
// %v = load(%ptr)            ->   %r = mul(%a, [%ptr])
// %r = mul(%a, %v)
//
// This is the transform x86 gets for free at instruction selection by
// having separate rr/rm instruction forms; with no instruction selection
// of our own, it belongs in a pass, positioned where LLVM's
// PeepholeOptimizer sits — after two-address lowering, before liveness.
//
// Requirements for a fold, all checked:
//   - the load defines a value with exactly one use in the whole IR
//   - the consumer declares that operand foldable (fold_read / fold_emit)
//     and has no folded operand yet (x86 allows one memory operand)
//   - load and consumer sit in the same region body, with no op that may
//     write memory between them, and no region op between them
//   - the load is unmasked and its base pointer is an IR value
struct FoldMemoryOperandsPass : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "FoldMemoryOperands"; }
};

// Transform: drop ops whose def nothing reads. LLVM's
// DeadMachineInstructionElim, scheduled where LLVM schedules its second
// run of it — immediately after the peephole pass, whose comment reads
// "Clean-up the dead code that may have been generated by peephole
// rewriting". Before liveness, so dead values never reach the allocator
// and never occupy a register.
//
// The deletion predicate follows MachineInstr::isDead: every def must be
// unused, and the op must be trivially dead otherwise
// (wouldBeTriviallyDead -> isSafeToMove, which rejects mayStore, calls,
// ordered/volatile accesses, terminators and unmodelled side effects).
// Two consequences worth spelling out, because both differ from the
// obvious guess:
//
//   - A dead *load* is deletable. LLVM only refuses ordered or volatile
//     ones, and deleting a load can only remove a fault, never add one.
//   - An op with no def is never deleted here, which is stricter than
//     LLVM. LLVM can delete a def-less instruction because flags and
//     memory are themselves modelled as defs, so liveness decides. This
//     IR models neither, so the DSL convention is that an op which
//     affects anything beyond its def declares no def at all (ir_use):
//     stores, compares and pointer advances all take that form, and
//     keeping every def-less op is what makes the convention safe.
//     Same role as LLVM's UnmodeledSideEffects, minus a flag nothing
//     would set.
//
// Uses are counted once and decremented as ops die, with the walk running
// backwards, so a chain of dead defs collapses in one sweep — LLVM visits
// blocks in post-order and instructions bottom-up for the same reason.
//
// The DSL needs this because it records operands a consumer may or may not
// use: `vlen::predicated` carries both a predicate and an element count,
// since interleaved stores on x86 cannot take a predicate and read the
// count instead. A body that only does plain loads and stores leaves the
// count dead.
struct DeadDefElimPass : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "DeadDefElim"; }
};

// Transform: insert explicit COPY ops before tied-operand instructions.
// LLVM's TwoAddressInstructionPass equivalent. Rewrites:
//   %result = FMA(%seed, %a, %b)  [tied_to=0]
// into:
//   %copy   = COPY(%seed)
//   %result = FMA(%copy, %a, %b)  [tied_to removed, emit assumes def==reads[0]]
// The coalescer (in RegisterAllocator) eliminates the COPY when possible.
struct TwoAddressPass : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "TwoAddress"; }
};

// Allocation: interference-based linear scan with integrated remat.
// May modify the IR (remat) and re-run LiveRangeAnalysis internally.
struct RegisterAllocator : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "RegisterAllocator"; }
};

// Verification: assert the invariants lowering depends on. Throws on
// violation — a failed verification means the kernel would be miscompiled.
struct VerifyPass : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "Verify"; }
};

// Debug: dump IR and allocation state.
struct DumpPass : IRPass {
    const char* label;
    explicit DumpPass(const char* l) : label(l) {}
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "Dump"; }
};

// Lowering: emit xbyak instructions using the assignment.
struct LoweringPass : IRPass {
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "Lowering"; }
};

// Build the default pass pipeline.
PassManager build_default_pipeline();

}  // namespace ov::intel_cpu::jit_kernel_ir

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Slice 1 of the jit_kernel register allocator — see jit_kernel_register_allocation.md
//
// Minimum viable IR + straight-line linear-scan allocator. No control flow,
// no spill, no DSL integration yet. This slice exists to prove the allocator
// core on hand-built IR with stub closures.
//
// Invariants preserved from day one (see design doc, "Invariants to preserve
// from day one"):
//   - SSA at the IR level: every Op.def is a fresh value id.
//   - Op is a plain struct with room to grow.
//   - Allocator is a swappable pass over (IR, pool) -> Assignment.
//   - Spill-victim is a swappable function (not exercised in Slice 1).
//   - Lowering is mechanical substitution only.

#pragma once

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

namespace ov::intel_cpu::jit_kernel_ir {

using value_id = std::uint32_t;
inline constexpr value_id invalid_value = std::numeric_limits<value_id>::max();

// LLVM-style register class. The allocator handles all classes in a single
// unified priority queue, picking from the correct physical pool per value.
// Class is stored per virtual register (on Op::def_rc / LiveRange::rc),
// NOT on PhysReg — matching LLVM's MachineRegisterInfo design.
enum class RegisterClass : std::uint8_t {
    Vec,   // Ymm/Zmm — SIMD register file
    GPR,   // Reg64 — general-purpose register file
};

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
struct EmitContext {
    std::optional<PhysReg> def;              // physical reg backing Op.def, if any
    const std::vector<PhysReg>& reads;       // physical regs backing Op.reads, in order
};

using EmitFn = std::function<void(const EmitContext&)>;

class IR;  // forward declaration for Op::body

// LLVM-style semantic opcode. The allocator ignores this — it only matters
// for IR transform passes (e.g., epilogue generation needs to distinguish
// loads/stores from pure math to replace them with partial variants).
enum class OpKind : std::uint8_t {
    Generic,  // math, constants, control flow — clone as-is in transforms
    Load,     // memory read  — epilogue replaces with partial/masked load
    Store,    // memory write — epilogue replaces with partial/masked store
};

// The one and only op type. The allocator reads only the dependency shape
// (reads/def/tied_to). OpKind is for transform passes.
struct Op {
    std::vector<value_id> reads;
    value_id def = invalid_value;        // invalid_value = no def (e.g. store)
    EmitFn emit;                         // called by lowering with allocated regs
    bool is_copy = false;                // trivial coalescing hint for pure copies
    int tied_to = -1;                    // LLVM-style tied operand: index into reads[]
                                         // that def must share a register with. -1 = none.
                                         // The allocator coalesces or inserts a copy.
    RegisterClass def_rc = RegisterClass::Vec;  // register class for the def value
    OpKind kind = OpKind::Generic;       // semantic tag for transform passes
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
    // OpKind tags the op for transform passes (epilogue generation etc.).
    value_id def(std::vector<value_id> reads, EmitFn emit, const char* name = "",
                 RegisterClass rc = RegisterClass::Vec,
                 OpKind kind = OpKind::Generic) {
        const value_id id = _next_value++;
        Op op;
        op.reads = std::move(reads);
        op.def = id;
        op.emit = std::move(emit);
        op.def_rc = rc;
        op.kind = kind;
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

    // Record an op that reads values but defines none (e.g. a store).
    void use(std::vector<value_id> reads, EmitFn emit, const char* name = "",
             OpKind kind = OpKind::Generic) {
        Op op;
        op.reads = std::move(reads);
        op.emit = std::move(emit);
        op.kind = kind;
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
        body_builder();
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

    [[nodiscard]] const std::list<Op>& ops() const noexcept { return _ops; }
    [[nodiscard]] std::list<Op>& ops() noexcept { return _ops; }
    [[nodiscard]] std::size_t value_count() const noexcept { return _next_value; }
    void set_value_count(value_id v) noexcept { _next_value = v; }

    void dump(std::ostream& os) const;

private:
    std::list<Op>& target() { return _cursor ? *_cursor : _ops; }

    std::list<Op> _ops;
    std::list<Op>* _cursor = nullptr;    // non-null = recording into a body
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

// Raised when the pool overflows. Slice 1 has no spill support — callers are
// expected to size the pool to fit the kernel, or the allocator fails fast.
// Spill support lands in Slice 4.
class allocation_failure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Pressure-repair pass: rewrites the IR by inserting one local single-use
// rematerialization when peak pressure exceeds the pool size for the value's
// register class.
// Returns true iff the IR was rewritten.
bool rematerialize_for_pressure(IR& ir,
                                const std::vector<LiveRange>& ranges,
                                std::uint32_t vec_pool_size,
                                std::uint32_t gpr_pool_size = 0);

// LLVM-style interference-based allocator with integrated remat.
// Returns Assignment on success. Returns std::nullopt when the IR was
// modified (a rematerializable value was cloned at each use site) — the
// caller should recompute live ranges and retry. Throws allocation_failure
// when no register is available and no rematerializable victim exists.
// gpr_pool_indices maps pool slots to physical GPR register indices.
std::optional<Assignment> linear_scan(IR& ir,
                                      std::vector<LiveRange>& ranges,
                                      std::uint32_t vec_pool_size,
                                      const std::vector<std::uint32_t>& gpr_pool_indices = {});

// Loop unrolling strategies.
enum class UnrollStrategy {
    none,       // no unrolling
    heuristic,  // LLVM-style: unroll_factor = min(trip_count, pool_size / body_pressure)
    feedback    // feedback-directed: trial allocation to find optimal factor
};

// IR transform pass: unroll loops in the IR.
// Clones loop body ops, remaps value_ids. Runs before allocation.
// Returns true if any loop was unrolled.
bool unroll_loops(IR& ir, std::uint32_t vec_pool_size, UnrollStrategy strategy = UnrollStrategy::heuristic);

// Test-only: remat a specific value — insert a clone (with reads
// preserved) before each use and rewrite reads.
// Returns true if the IR was modified.
bool unit_test_api_remat_value(IR& ir, value_id vid);

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
    // Per-class register pools. LLVM-style: one allocator, class-specific pools.
    // Vec pool is contiguous 0..vec_pool_size-1 (physical index == pool slot).
    // GPR pool uses an indirection table (gpr_pool_indices) because allocable
    // GPR indices are non-contiguous (rsp, rbp, abi_param excluded).
    // Mirrors LLVM's AllocationOrder (RegisterClassInfo::getOrder).
    std::uint32_t vec_pool_size = 0;
    std::vector<std::uint32_t> gpr_pool_indices;  // allocable GPR register indices

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

// Transform: loop unrolling (IR rewrite).
struct LoopUnrollPass : IRPass {
    UnrollStrategy strategy;
    explicit LoopUnrollPass(UnrollStrategy s = UnrollStrategy::none) : strategy(s) {}
    bool run(IR& ir, PassContext& ctx) override;
    const char* name() const override { return "LoopUnroll"; }
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
PassManager build_default_pipeline(UnrollStrategy unroll = UnrollStrategy::none);

}  // namespace ov::intel_cpu::jit_kernel_ir

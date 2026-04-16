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

// A lightweight physical register handle. In Slice 1 this is just an index
// into a generic pool; later slices will specialize per-pool (vector / GPR /
// mask). The allocator is pool-agnostic.
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

// The one and only op type. No taxonomy. The allocator never inspects the
// emit closure — it only reads the dependency shape.
struct Op {
    std::vector<value_id> reads;
    value_id def = invalid_value;        // invalid_value = no def (e.g. store)
    EmitFn emit;                         // called by lowering with allocated regs
    bool is_copy = false;                // trivial coalescing hint (unused in Slice 1)
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
    value_id def(std::vector<value_id> reads, EmitFn emit, const char* name = "") {
        const value_id id = _next_value++;
        target().push_back(Op{std::move(reads), id, std::move(emit), /*is_copy=*/false, /*body=*/nullptr, /*is_loop=*/false, name});
        return id;
    }

    // Record an op that reads values but defines none (e.g. a store).
    void use(std::vector<value_id> reads, EmitFn emit, const char* name = "") {
        target().push_back(Op{std::move(reads), invalid_value, std::move(emit), /*is_copy=*/false, /*body=*/nullptr, /*is_loop=*/false, name});
    }

    // Record a copy-like op: defines a fresh value whose contents come from a
    // single source. Flagged for the allocator's trivial coalescing pass.
    value_id copy(value_id src, EmitFn emit, const char* name = "") {
        const value_id id = _next_value++;
        target().push_back(Op{std::vector<value_id>{src}, id, std::move(emit), /*is_copy=*/true, /*body=*/nullptr, /*is_loop=*/false, name});
        return id;
    }

    // Record a region op — a nested body of ops. The emit closure is
    // called at lowering time before the body ops are lowered.
    // `is_loop` controls whether intervals are extended across the body.
    template <typename BodyBuilder>
    void region(EmitFn emit, BodyBuilder&& body_builder, bool is_loop = false) {
        auto body = std::make_unique<IR>();
        // Save/restore cursor so nested regions (e.g. ir_if inside a loop)
        // don't clobber the outer cursor.
        auto* saved_cursor = _cursor;
        _cursor = &body->_ops;
        body_builder();
        _cursor = saved_cursor;

        target().push_back(Op{{}, invalid_value, std::move(emit), false,
                              std::move(body), is_loop});
    }

    // Convenience: record a loop region (intervals extended across body).
    template <typename BodyBuilder>
    void loop(EmitFn emit, BodyBuilder&& body_builder) {
        region(std::move(emit), std::forward<BodyBuilder>(body_builder), /*is_loop=*/true);
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

// A contiguous [start, end] range within a LiveRange. LLVM naming.
struct Segment {
    std::uint32_t start = 0;
    std::uint32_t end = 0;
};

// Live range for a single value id — a sorted list of non-overlapping
// Segments. Replaces the old flat Interval. LLVM naming.
//
// For straight-line code each value has one segment. Sibling branch bodies
// (then/else) produce separate segments for the same value, preserving
// per-branch liveness information.
struct LiveRange {
    value_id id = invalid_value;
    std::vector<Segment> segments;  // sorted by start, non-overlapping
    value_id copy_of = invalid_value;  // source value if this is a copy op

    // First segment's start. Returns max uint32 if empty (not yet defined).
    [[nodiscard]] std::uint32_t beginIndex() const noexcept;
    // Last segment's end. Returns 0 if empty.
    [[nodiscard]] std::uint32_t endIndex() const noexcept;
    // True if `index` falls within any segment.
    [[nodiscard]] bool liveAt(std::uint32_t index) const noexcept;
    // Insert a segment, maintaining sorted order. Merges overlapping
    // segments but NOT merely adjacent ones (preserves branch boundaries).
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
// rematerialization when peak pressure exceeds `pool_size`.
// Returns true iff the IR was rewritten.
bool rematerialize_for_pressure(IR& ir,
                                const std::vector<LiveRange>& ranges,
                                std::uint32_t pool_size);

// Pass 3: plain linear scan (Poletto & Sarkar 1999) with trivial coalescing.
// Assumes any optional pressure-repair pass already ran.
Assignment linear_scan(IR& ir,
                       std::vector<LiveRange>& ranges,
                       std::uint32_t pool_size);

// Debug helper: text dump of the op stream. Used by IR::dump and by tests
// to eyeball what was recorded.
void dump_ops(std::ostream& os, const IR& ir);

// Debug helper: dump live ranges and their assigned physical registers.
// Prints "id: [start, end] ... -> pN" per value. If a value is missing from
// the assignment map (e.g. because allocation failed before it was processed),
// prints "unassigned".
void dump_assignment(std::ostream& os,
                     const std::vector<LiveRange>& ranges,
                     const Assignment& assignment);

}  // namespace ov::intel_cpu::jit_kernel_ir

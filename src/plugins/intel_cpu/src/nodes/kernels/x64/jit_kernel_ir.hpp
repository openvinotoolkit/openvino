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
};

// Builder / container for the recorded op stream.
// Ops record into the current insertion target (top-level by default).
// Loop ops contain a nested IR (body) — like MLIR's regions.
class IR {
public:
    // Record an op that defines a fresh value. Returns the new value id.
    value_id def(std::vector<value_id> reads, EmitFn emit) {
        const value_id id = _next_value++;
        target().push_back(Op{std::move(reads), id, std::move(emit), /*is_copy=*/false, /*body=*/nullptr, /*is_loop=*/false});
        return id;
    }

    // Record an op that reads values but defines none (e.g. a store).
    void use(std::vector<value_id> reads, EmitFn emit) {
        target().push_back(Op{std::move(reads), invalid_value, std::move(emit), /*is_copy=*/false, /*body=*/nullptr, /*is_loop=*/false});
    }

    // Record a copy-like op: defines a fresh value whose contents come from a
    // single source. Flagged for the allocator's trivial coalescing pass.
    value_id copy(value_id src, EmitFn emit) {
        const value_id id = _next_value++;
        target().push_back(Op{std::vector<value_id>{src}, id, std::move(emit), /*is_copy=*/true, /*body=*/nullptr, /*is_loop=*/false});
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

// Live interval for a single value id. Half-open convention: the value is
// live at op indices [start, end] inclusive.
//   - start = op index where the value is defined.
//   - end   = op index of the last read, or start if the value is never read.
struct Interval {
    value_id id = invalid_value;
    std::uint32_t start = 0;
    std::uint32_t end = 0;
    value_id copy_of = invalid_value;  // source value if this is a copy op

    [[nodiscard]] bool covers(std::uint32_t op_index) const noexcept {
        return op_index >= start && op_index <= end;
    }
};

// Interval construction. Walks the op tree recursively — loop bodies are
// traversed inline and intervals of values live inside a loop are extended
// to cover the full loop body.
std::vector<Interval> compute_intervals(const IR& ir);

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

// Pass 3: linear scan (Poletto & Sarkar 1999) with trivial coalescing
// and rematerialization. When the pool overflows, evicts a rematerializable
// value (empty reads) and clones its def op before the next use.
// Mutates `ir` to insert remat ops.
Assignment linear_scan(IR& ir,
                       std::vector<Interval>& intervals,
                       std::uint32_t pool_size);

// Debug helper: text dump of the op stream. Used by IR::dump and by tests
// to eyeball what was recorded.
void dump_ops(std::ostream& os, const IR& ir);

// Debug helper: dump intervals and their assigned physical registers.
// Prints "id: [start, end] -> pN" per value. If a value is missing from the
// assignment map (e.g. because allocation failed before it was processed),
// prints "unassigned".
void dump_assignment(std::ostream& os,
                     const std::vector<Interval>& intervals,
                     const Assignment& assignment);

}  // namespace ov::intel_cpu::jit_kernel_ir

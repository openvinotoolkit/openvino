// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit tests for jit_kernel_ir.hpp.
//
// Coverage:
//   - Live range construction on hand-built IR: straight-line, branches,
//     loops, half-open segments, early/late slots, re-defs.
//   - Allocation: register reuse across disjoint ranges, coalescing of
//     copies and tied operands, mixed Vec/GPR classes over two pools,
//     rematerialization, pool exhaustion (there is no spiller, so
//     exhaustion throws allocation_failure).
//   - Verifier: rejects interference violations, missing assignments,
//     dangling reads and out-of-pool registers.
//   - Randomized allocation: fuzzed IR must always satisfy the verifier.
//   - End-to-end IR mode: record through DSL operators, allocate, lower,
//     execute the generated kernel and check results — including
//     differential tests against a scalar reference over random widths.

#include <gtest/gtest.h>
#include <kernels/x64/jit_kernel.hpp>
#include <kernels/x64/jit_kernel_ir.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ov::intel_cpu;
using namespace ov::intel_cpu::jit_kernel_ir;

namespace {

// IR mode is always active — no env var needed.

EmitFn stub() {
    return [](const EmitContext&) {};
}

// Helper: total op count including nested region bodies.
std::size_t count_ops_recursive(const std::list<Op>& ops) {
    std::size_t n = 0;
    for (const auto& op : ops) {
        ++n;
        if (op.body) {
            n += count_ops_recursive(op.body->ops());
        }
    }
    return n;
}

// Helper: assign_registers with an explicit pool configuration.
std::optional<Assignment> assign_registers_test(IR& ir, std::vector<LiveRange>& ranges,
                                           std::uint32_t vec_pool_size,
                                           const std::vector<std::uint32_t>& gpr_pool = {}) {
    PassContext ctx;
    for (std::uint32_t i = 0; i < vec_pool_size; ++i) {
        ctx.vec_pool_indices.push_back(i);
    }
    ctx.gpr_pool_indices = gpr_pool;
    return assign_registers(ir, ranges, ctx);
}

// Helper: assert every value that appears in `intervals` (with a real start)
// has been assigned a physical register in `assignment`.
void expect_all_assigned(const std::vector<LiveRange>& ranges, const Assignment& assignment) {
    for (const auto& lr : ranges) {
        if (lr.empty()) {
            continue;
        }
        EXPECT_TRUE(assignment.reg.count(lr.id) == 1)
            << "value %" << lr.id << " was not assigned a register";
    }
}

// Helper: assert no two intervals that overlap in op-index space landed on
// the same physical register. This is the core correctness property of the
// allocator — anything else is secondary.
// Use LiveRange::overlaps — LLVM-style method.
bool segments_overlap(const LiveRange& a, const LiveRange& b) {
    return a.overlaps(b);
}

void expect_no_overlap_conflict(const std::vector<LiveRange>& ranges,
                                const Assignment& assignment) {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto& a = ranges[i];
        if (a.empty()) {
            continue;
        }
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            const auto& b = ranges[j];
            if (b.empty()) {
                continue;
            }
            // Segment-level overlap: two values can share a register if
            // their segments never overlap (e.g., different branches).
            if (!segments_overlap(a, b)) {
                continue;
            }
            const auto ra = assignment.reg.find(a.id);
            const auto rb = assignment.reg.find(b.id);
            if (ra == assignment.reg.end() || rb == assignment.reg.end()) {
                continue;
            }
            EXPECT_NE(ra->second, rb->second)
                << "overlapping live ranges %" << a.id << " and %" << b.id
                << " share physical register p" << ra->second.idx;
        }
    }
}

std::size_t count_reads_recursive(const std::list<Op>& ops, value_id id) {
    std::size_t count = 0;
    for (const auto& op : ops) {
        if (op.body) {
            count += count_reads_recursive(op.body->ops(), id);
            continue;
        }
        for (auto read : op.reads) {
            if (read == id) {
                ++count;
            }
        }
    }
    return count;
}

}  // namespace

TEST(JitKernelIR, LiveRangesOnStraightLine) {
    // Linear chain: a, b, c = a + b, d = c + a.
    // - a defined at op 0, last read at op 3 (in d's definition).
    // - b defined at op 1, last read at op 2.
    // - c defined at op 2, last read at op 3.
    // - d defined at op 3, never read: end == start == 3.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    const value_id d = ir.def({c, a}, stub());

    auto ranges = compute_live_ranges(ir);
    ASSERT_EQ(ranges.size(), 4U);

    // Straight-line code: one segment per value.
    EXPECT_EQ(ranges[a].segments.size(), 1U);
    // Early/late slots: def at late(N)=2N+1, read at early(N)=2N, half-open end=early+1.
    EXPECT_EQ(ranges[a].beginIndex(), 1U);   // def at late(0)
    EXPECT_EQ(ranges[a].endIndex(), 7U);     // last read at early(3), half-open end=7
    EXPECT_EQ(ranges[b].beginIndex(), 3U);   // def at late(1)
    EXPECT_EQ(ranges[b].endIndex(), 5U);     // last read at early(2), half-open end=5
    EXPECT_EQ(ranges[c].beginIndex(), 5U);   // def at late(2)
    EXPECT_EQ(ranges[c].endIndex(), 7U);     // last read at early(3), half-open end=7
    EXPECT_EQ(ranges[d].beginIndex(), 7U);   // def at late(3)
    EXPECT_EQ(ranges[d].endIndex(), 8U);     // never read, half-open end=8
}

TEST(JitKernelIR, LinearScanFitsInPool) {
    // Same 4-op chain. With early/late slots, peak overlap is 2
    // (reads at early slots don't overlap with defs at late slots).
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    const value_id d = ir.def({c, a}, stub());
    (void)d;

    auto ranges = compute_live_ranges(ir);
    const auto assignment = *assign_registers_test(ir, ranges, /*pool_size=*/4);

    expect_all_assigned(ranges, assignment);
    expect_no_overlap_conflict(ranges, assignment);
    EXPECT_EQ(assignment.peak_live, 2U);
}

TEST(JitKernelIR, LinearScanReusesFreedRegisters) {
    // Build a long chain of six values where each value dies before the next
    // fresh def begins. The allocator should recycle a single register across
    // the entire chain, yielding peak_live == 2 (old + new transient window).
    //
    //   %0 = def              // live [0, 1]
    //   %1 = op(%0)           // live [1, 2]  (def), %0 dies after op 1
    //   %2 = op(%1)           // live [2, 3]
    //   %3 = op(%2)           // live [3, 4]
    //   %4 = op(%3)           // live [4, 5]
    //   %5 = op(%4)           // live [5, 5]  (never read)
    IR ir;
    value_id prev = ir.def({}, stub());
    for (int i = 0; i < 5; ++i) {
        prev = ir.def({prev}, stub());
    }
    (void)prev;

    auto ranges = compute_live_ranges(ir);
    const auto assignment = *assign_registers_test(ir, ranges, /*pool_size=*/2);

    expect_all_assigned(ranges, assignment);
    expect_no_overlap_conflict(ranges, assignment);
    EXPECT_LE(assignment.peak_live, 2U);
    EXPECT_GE(assignment.peak_live, 1U);
}

TEST(JitKernelIR, AssignmentThrowsOnPoolOverflow) {
    // Five long-lived values all bundled into one final op. The first is
    // remat-able but doesn't interfere with the failing value (v4), so
    // the allocator correctly throws rather than entering an infinite
    // remat loop.
    auto build_chain = [](IR& ir) {
        value_id prev = ir.def({}, stub());
        std::vector<value_id> values;
        values.push_back(prev);
        for (int i = 1; i < 5; ++i) {
            prev = ir.def({prev}, stub());
            values.push_back(prev);
        }
        ir.use(values, stub());
    };

    // Pool of 4 must fail — peak is 5. The allocator may attempt remat
    // but can't reduce the 5-value peak at the final use op. Bound
    // retries to prevent infinite remat chains.
    IR ir;
    build_chain(ir);
    auto ranges = compute_live_ranges(ir);
    EXPECT_THROW({
        for (std::size_t attempt = 0, limit = ranges.size(); attempt < limit; ++attempt) {
            auto result = assign_registers_test(ir, ranges, /*pool_size=*/4);
            if (result) break;
            ranges = compute_live_ranges(ir);
        }
        throw allocation_failure("remat exhausted without progress");
    }, allocation_failure);

    // Same chain with a pool large enough should succeed.
    IR ir2;
    build_chain(ir2);
    auto ranges2 = compute_live_ranges(ir2);
    auto result2 = assign_registers_test(ir2, ranges2, /*pool_size=*/5);
    ASSERT_TRUE(result2.has_value());
    expect_all_assigned(ranges2, *result2);
    expect_no_overlap_conflict(ranges2, *result2);
    EXPECT_EQ(result2->peak_live, 5U);
}

TEST(JitKernelIR, UseWithoutDefDoesNotCreateValue) {
    // use() records an op with no def — verifies ir.value_count() doesn't
    // grow for uses, only for defs. Important for peak_live accounting: a
    // store at the end of a kernel must not be counted as a fresh live value.
    IR ir;
    const value_id v = ir.def({}, stub());
    ir.use({v}, stub());

    EXPECT_EQ(ir.value_count(), 1U);
    EXPECT_EQ(ir.ops().size(), 2U);

    auto ranges = compute_live_ranges(ir);
    ASSERT_EQ(ranges.size(), 1U);
    EXPECT_EQ(ranges[v].beginIndex(), 1U);   // def at late(0)
    EXPECT_EQ(ranges[v].endIndex(), 3U);    // read at early(1), half-open end=3

    const auto assignment = *assign_registers_test(ir, ranges, /*pool_size=*/1);
    EXPECT_EQ(assignment.peak_live, 1U);
}

TEST(JitKernelIR, CopyOpRecordedAsCopy) {
    // copy() sets is_copy=true on the recorded Op. Slice 1 does not use this
    // flag for coalescing, but the recording path must be correct so Slice 4
    // can consume it.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.copy(a, stub());

    ASSERT_EQ(ir.ops().size(), 2U);
    auto it = ir.ops().begin();
    EXPECT_FALSE(it->is_copy);
    ++it;
    EXPECT_TRUE(it->is_copy);
    EXPECT_EQ(it->def, b);
    ASSERT_EQ(it->reads.size(), 1U);
    EXPECT_EQ(it->reads[0], a);
}

TEST(JitKernelIR, DumpProducesNonEmptyText) {
    // Smoke test for the dump helpers. The exact format is not specified —
    // the tests only require that the dump is non-empty and includes some
    // structural tokens so future refactors don't silently break the dump
    // path.
    IR ir;
    const value_id a = ir.def({}, stub());
    const value_id b = ir.def({}, stub());
    const value_id c = ir.def({a, b}, stub());
    (void)c;

    auto ranges = compute_live_ranges(ir);
    const auto assignment = *assign_registers_test(ir, ranges, /*pool_size=*/3);

    std::ostringstream op_dump;
    dump_ops(op_dump, ir);
    EXPECT_NE(op_dump.str().find('%'), std::string::npos);
    EXPECT_NE(op_dump.str().find("op"), std::string::npos);

    std::ostringstream assign_dump;
    dump_assignment(assign_dump, ranges, assignment);
    EXPECT_NE(assign_dump.str().find('%'), std::string::npos);
    EXPECT_NE(assign_dump.str().find("peak_live"), std::string::npos);
}

TEST(JitKernelIR, DeadValueFreesRegisterImmediately) {
    // A value that is defined but never read should have its register
    // freed after the def op, so a subsequent def can reuse it with a
    // pool of size 1.
    IR ir;
    (void)ir.def({}, stub());  // dead
    (void)ir.def({}, stub());  // dead, would fail on pool=1 if dead value held its reg

    auto ranges = compute_live_ranges(ir);
    const auto assignment = *assign_registers_test(ir, ranges, /*pool_size=*/1);
    EXPECT_EQ(assignment.peak_live, 1U);
    EXPECT_EQ(assignment.reg.size(), 2U);
}

TEST(JitKernelIR, AllocatorRematerializesUnderPressure) {
    // %c0, %hold and %pressure are live at the same point, so a pool of two
    // cannot hold them. %c0 has no reads, hence is rematerializable: the
    // allocator clones it at its uses, reports "IR modified" (nullopt) and
    // the caller re-runs analysis.
    IR ir;
    const value_id c0 = ir.def({}, stub());       // remat-able
    const value_id hold = ir.def({c0}, stub());   // long-lived, not remat-able
    const value_id pressure = ir.def({hold}, stub());
    (void)pressure;

    ir.region(stub(), [&]() {
        ir.use({c0}, stub());
    });
    ir.region(stub(), [&]() {
        ir.use({c0}, stub());
    });
    ir.use({hold}, stub());  // keep `hold` live past both regions

    auto ranges = compute_live_ranges(ir);
    const auto values_before = ir.value_count();

    const auto first = assign_registers_test(ir, ranges, /*pool_size=*/2);
    EXPECT_FALSE(first.has_value()) << "expected remat (IR rewritten), not an assignment";
    EXPECT_GT(ir.value_count(), values_before) << "remat should have introduced clones";
    EXPECT_LT(count_reads_recursive(ir.ops(), c0), 2U)
        << "uses of the victim should now read clones";

    // Retry loop, as RegisterAllocator does, must converge.
    for (int attempt = 0; attempt < 8; ++attempt) {
        ranges = compute_live_ranges(ir);
        const auto result = assign_registers_test(ir, ranges, /*pool_size=*/2);
        if (result) {
            expect_no_overlap_conflict(ranges, *result);
            return;
        }
    }
    FAIL() << "allocation did not converge after remat";
}

TEST(JitKernelIR, InputAwareRematClonesWithReads) {
    // Verify that remat of a value with inputs produces clones that
    // carry the victim's reads (not empty). This is critical: the emit
    // closure reads physical registers from EmitContext.reads, so the
    // clone Op must list the same value_id reads as the original def.
    //
    //   %c0 = def()              — constant
    //   %c1 = def()              — constant
    //   %derived = def({c0, c1}) — two-input op
    //   use({derived})           — single use
    //   use({c0, c1})            — keeps constants alive

    IR ir;
    const auto c0 = ir.def({}, stub());
    const auto c1 = ir.def({}, stub());
    const auto derived = ir.def({c0, c1}, stub());
    ir.use({derived}, stub());
    ir.use({c0, c1}, stub());

    // Force remat of "derived" and verify the clone's reads.
    EXPECT_TRUE(unit_test_api_remat_value(ir, derived));

    // The original use of "derived" was rewritten to read a clone.
    // Find the clone and verify it has reads = {c0, c1}.
    bool found_clone_with_reads = false;
    for (const auto& op : ir.ops()) {
        if (std::string(op.name) == "remat") {
            EXPECT_EQ(op.reads.size(), 2U)
                << "remat clone must carry the victim's input reads";
            if (op.reads.size() == 2U) {
                EXPECT_EQ(op.reads[0], c0);
                EXPECT_EQ(op.reads[1], c1);
            }
            found_clone_with_reads = true;
        }
    }
    EXPECT_TRUE(found_clone_with_reads)
        << "remat should have inserted a clone";

    // After remat, allocation should succeed with a tight pool.
    // "derived" is now dead (range = [def, def]). The clone is
    // short-lived (right before the use). Pool = 3 should suffice
    // for c0, c1, and the clone (derived is dead).
    auto ranges = compute_live_ranges(ir);
    auto result = assign_registers_test(ir, ranges, /*pool_size=*/3);
    ASSERT_TRUE(result.has_value());
    expect_all_assigned(ranges, *result);
    expect_no_overlap_conflict(ranges, *result);
}

TEST(JitKernelIR, InputAwareRematReducesPressure) {
    // Verify that rematerializing a derived value whose inputs are
    // already live reduces pressure in the middle of the IR.
    //
    //   %c0 = def()              [0, 6]
    //   %c1 = def()              [1, 6]
    //   %derived = def({c0,c1})  [2, 5]  — holds register across pressure window
    //   %t0 = def()              [3, 4]
    //   %t1 = def({t0})          [4, 5]
    //   use({derived, t1})       [5]
    //   use({c0, c1})            [6]     — keeps constants alive
    //
    // Peak at 4: c0, c1, derived, t0, t1 = 5. Pool = 4.
    // After remat of derived: clone at use site, derived dead.
    // Peak at 4: c0, c1, t0, t1 = 4. Fits.

    IR ir;
    const auto c0 = ir.def({}, stub());
    const auto c1 = ir.def({}, stub());
    const auto derived = ir.def({c0, c1}, stub());
    const auto t0 = ir.def({}, stub());
    const auto t1 = ir.def({t0}, stub());
    ir.use({derived, t1}, stub());
    ir.use({c0, c1}, stub());

    // Directly remat "derived" (input-aware: c0, c1 outlive it).
    EXPECT_TRUE(unit_test_api_remat_value(ir, derived));

    // Allocation should now succeed with pool=4.
    auto ranges = compute_live_ranges(ir);
    auto result = assign_registers_test(ir, ranges, /*pool_size=*/4);
    ASSERT_TRUE(result.has_value()) << "allocation should succeed after remat";
    expect_all_assigned(ranges, *result);
    expect_no_overlap_conflict(ranges, *result);
    EXPECT_LE(result->peak_live, 4U);
}

// ── LiveRange / Segment unit tests ─────────────────────────────────────

TEST(JitKernelIR, LiveRangeAddSegmentMergesOverlapping) {
    LiveRange lr;
    lr.id = 0;
    lr.addSegment({5, 10});
    lr.addSegment({0, 3});
    EXPECT_EQ(lr.segments.size(), 2U);  // [0,3] [5,10] — not adjacent-merged

    lr.addSegment({3, 5});  // overlaps both: bridges the gap
    EXPECT_EQ(lr.segments.size(), 1U);  // [0,10]
    EXPECT_EQ(lr.segments[0].start, 0U);
    EXPECT_EQ(lr.segments[0].end, 10U);
}

TEST(JitKernelIR, LiveRangeAddSegmentKeepsAdjacentSeparate) {
    LiveRange lr;
    lr.id = 0;
    lr.addSegment({0, 2});
    lr.addSegment({3, 5});
    // Adjacent but NOT overlapping — must stay separate (branch boundary).
    EXPECT_EQ(lr.segments.size(), 2U);
    EXPECT_EQ(lr.segments[0].start, 0U);
    EXPECT_EQ(lr.segments[0].end, 2U);
    EXPECT_EQ(lr.segments[1].start, 3U);
    EXPECT_EQ(lr.segments[1].end, 5U);
}

TEST(JitKernelIR, LiveRangeLiveAt) {
    LiveRange lr;
    lr.id = 0;
    lr.addSegment({2, 5});
    lr.addSegment({8, 10});
    EXPECT_FALSE(lr.liveAt(1));
    EXPECT_TRUE(lr.liveAt(2));
    EXPECT_TRUE(lr.liveAt(3));
    EXPECT_FALSE(lr.liveAt(5));   // half-open: [2,5) excludes 5
    EXPECT_FALSE(lr.liveAt(6));
    EXPECT_FALSE(lr.liveAt(7));
    EXPECT_TRUE(lr.liveAt(8));
    EXPECT_FALSE(lr.liveAt(10));  // half-open: [8,10) excludes 10
    EXPECT_FALSE(lr.liveAt(11));
}

TEST(JitKernelIR, LiveRangesPerBranchSegments) {
    // Value %a defined before branches, used only in then-branch.
    // Value %b defined before branches, used only in else-branch.
    // Each gets a contiguous segment from def through its branch use
    // (the value must survive in its register from def to use).
    // Region ops own an index too (they emit the branch), so indices are:
    //   0: def a, 1: def b, 2: region, 3: use a, 4: region, 5: use b
    IR ir;
    const value_id a = ir.def({}, stub());     // index 0
    const value_id b = ir.def({}, stub());     // index 1

    ir.region(stub(), [&]() {                  // index 2 (then-branch)
        ir.use({a}, stub());                   // index 3
    });
    ir.region(stub(), [&]() {                  // index 4 (else-branch)
        ir.use({b}, stub());                   // index 5
    });

    auto ranges = compute_live_ranges(ir);

    // %a: def at late(0)=1, use at early(3)=6 → [1, 7).
    EXPECT_EQ(ranges[a].segments.size(), 1U);
    EXPECT_EQ(ranges[a].beginIndex(), 1U);
    EXPECT_EQ(ranges[a].endIndex(), 7U);
    EXPECT_FALSE(ranges[a].liveAt(0));
    EXPECT_TRUE(ranges[a].liveAt(1));
    EXPECT_TRUE(ranges[a].liveAt(6));
    EXPECT_FALSE(ranges[a].liveAt(7));

    // %b: def at late(1)=3, use at early(5)=10 → [3, 11). It stays live
    // across the first region: the then-branch may be taken and %b is
    // needed afterwards either way.
    EXPECT_EQ(ranges[b].segments.size(), 1U);
    EXPECT_EQ(ranges[b].beginIndex(), 3U);
    EXPECT_EQ(ranges[b].endIndex(), 11U);
    EXPECT_FALSE(ranges[b].liveAt(2));
    EXPECT_TRUE(ranges[b].liveAt(3));
    EXPECT_TRUE(ranges[b].liveAt(10));
    EXPECT_FALSE(ranges[b].liveAt(11));
}

TEST(JitKernelIR, LiveRangesUsedInBothBranches) {
    // Value defined before branches, used in both. The parent segment
    // extends through both branch uses, merging into one contiguous range.
    IR ir;
    const value_id a = ir.def({}, stub());     // index 0

    ir.region(stub(), [&]() {                  // index 1
        ir.use({a}, stub());                   // index 2
    });
    ir.region(stub(), [&]() {                  // index 3
        ir.use({a}, stub());                   // index 4
    });

    auto ranges = compute_live_ranges(ir);

    // %a: def at late(0)=1, last use at early(4)=8 → single segment [1, 9).
    EXPECT_EQ(ranges[a].segments.size(), 1U);
    EXPECT_EQ(ranges[a].beginIndex(), 1U);
    EXPECT_EQ(ranges[a].endIndex(), 9U);
}

TEST(JitKernelIR, LiveRangesUsedAfterBranch) {
    // Value used before and after branches — top-level segment spans
    // the branch region, merging with branch-body segments.
    IR ir;
    const value_id a = ir.def({}, stub());     // index 0

    ir.region(stub(), [&]() {                  // index 1
        ir.use({a}, stub());                   // index 2
    });
    ir.region(stub(), [&]() {                  // index 3
        ir.use({a}, stub());                   // index 4
    });

    ir.use({a}, stub());                       // index 5

    auto ranges = compute_live_ranges(ir);

    // %a: def at late(0)=1, last read at early(5)=10 → one segment [1, 11).
    EXPECT_EQ(ranges[a].segments.size(), 1U);
    EXPECT_EQ(ranges[a].beginIndex(), 1U);
    EXPECT_EQ(ranges[a].endIndex(), 11U);
}

// ── Loop-carried accumulators ──────────────────────────────────────────
//
// def_into() redefines an existing value instead of creating a new one,
// which is how an accumulator survives a back edge. The two properties
// that matter: the value stays live across the whole loop (so nothing
// else may take its register), and it keeps one register rather than
// being copied per iteration.

TEST(JitKernelIR, AccumulatorStaysLiveAcrossTheLoop) {
    IR ir;
    const value_id acc = ir.def({}, stub(), "zero");     // 0: before the loop
    const value_id idx = ir.def({}, stub(), "idx", RegisterClass::GPR);  // 1

    ir.loop({idx}, stub(), [&] {                          // 2: header
        const value_id x = ir.def({}, stub(), "load");    // 3
        ir.def_into(acc, {acc, x}, stub(), "accumulate"); // 4
    });
    ir.use({acc}, stub(), "store");                       // 5

    auto ranges = compute_live_ranges(ir);

    // Live from its definition through the loop to the store: a hole
    // anywhere here would let the allocator reuse the register mid-loop.
    EXPECT_EQ(ranges[acc].beginIndex(), 1U);
    EXPECT_GE(ranges[acc].endIndex(), 10U) << "must reach the store at op 5";

    // Two defs of one value, which is the whole point.
    std::size_t def_count = 0;
    for (const auto& op : ir.ops()) {
        def_count += (op.def == acc) ? 1 : 0;
        if (op.body) {
            for (const auto& inner : op.body->ops()) {
                def_count += (inner.def == acc) ? 1 : 0;
            }
        }
    }
    EXPECT_EQ(def_count, 2U);
}

TEST(JitKernelIR, AccumulatorKeepsOneRegisterAndNeedsNoCopy) {
    IR ir;
    const value_id acc = ir.def({}, stub(), "zero");
    const value_id idx = ir.def({}, stub(), "idx", RegisterClass::GPR);
    ir.loop({idx}, stub(), [&] {
        const value_id x = ir.def({}, stub(), "load");
        ir.def_into(acc, {acc, x}, stub(), "accumulate");
    });
    ir.use({acc}, stub(), "store");

    auto ranges = compute_live_ranges(ir);
    auto assignment = assign_registers_test(ir, ranges, /*vec_pool_size=*/4, {0, 1});
    ASSERT_TRUE(assignment.has_value());

    // TwoAddressPass inserts a copy for a *tied* operand; def_into needs
    // none, because the op already reads the value it defines.
    PassContext ctx;
    TwoAddressPass two_address;
    EXPECT_FALSE(two_address.run(ir, ctx)) << "no tie left to lower";
}

// Several accumulators at once, which is the brgemm shape: a register
// tile live across the whole loop nest. Checks the allocator keeps them
// distinct rather than reusing a register that is still accumulating.
TEST(JitKernelIR, ManyAccumulatorsGetDistinctRegisters) {
    constexpr std::uint32_t tile = 12;
    IR ir;
    std::vector<value_id> acc;
    acc.reserve(tile);
    for (std::uint32_t i = 0; i < tile; ++i) {
        acc.push_back(ir.def({}, stub(), "zero"));
    }
    const value_id idx = ir.def({}, stub(), "idx", RegisterClass::GPR);

    ir.loop({idx}, stub(), [&] {
        const value_id b = ir.def({}, stub(), "load_b");
        for (auto a : acc) {
            ir.def_into(a, {a, b}, stub(), "accumulate");
        }
    });
    for (auto a : acc) {
        ir.use({a}, stub(), "store");
    }

    auto ranges = compute_live_ranges(ir);
    auto assignment = assign_registers_test(ir, ranges, tile + 2, {0, 1});
    ASSERT_TRUE(assignment.has_value()) << "tile of " << tile << " did not fit";

    std::set<std::uint32_t> used;
    for (auto a : acc) {
        used.insert(assignment->reg.at(a).idx);
    }
    EXPECT_EQ(used.size(), tile) << "accumulators must not share registers";
}

// ── Memory operand folding ─────────────────────────────────────────────

namespace {

// A load: reads a base pointer, defines a value, declares its memory
// effects so the pass can reason about ordering.
value_id record_load(IR& ir, value_id base, std::uint32_t offset = 0) {
    const auto vid = ir.def({base}, stub(), "load");
    auto& op = ir.last();
    op.may_load = true;
    op.mem_ptr_read = 0;
    op.mem_offset = offset;
    return vid;
}

// A splat-from-memory: same shape as a load, different memory form.
value_id record_broadcast(IR& ir, value_id base, std::uint32_t offset = 0) {
    const auto vid = record_load(ir, base, offset);
    ir.last().load_form = fold_form::element_broadcast;
    return vid;
}

// A consumer with `foldable` as its bitmask of memory-capable operands.
value_id record_consumer(IR& ir, std::vector<value_id> reads, std::uint8_t foldable) {
    const auto vid = ir.def(std::move(reads), stub(), "consume");
    auto& op = ir.last();
    op.foldable_reads = foldable;
    op.fold_emit = stub();
    return vid;
}

// A consumer that also accepts the broadcast form on `broadcastable`.
value_id record_broadcast_consumer(IR& ir, std::vector<value_id> reads, std::uint8_t foldable,
                                   std::uint8_t broadcastable) {
    const auto vid = record_consumer(ir, std::move(reads), foldable);
    auto& op = ir.last();
    op.broadcast_foldable_reads = broadcastable;
    op.broadcast_fold_emit = stub();
    return vid;
}

bool has_load(const std::list<Op>& ops) {
    for (const auto& op : ops) {
        if (op.may_load && op.def != invalid_value && op.folded_read < 0) {
            return true;
        }
    }
    return false;
}

bool run_folding(IR& ir) {
    PassContext ctx;
    FoldMemoryOperandsPass pass;
    return pass.run(ir, ctx);
}

}  // namespace

// ── Dead def elimination ───────────────────────────────────────────────

namespace {

bool run_dce(IR& ir) {
    PassContext ctx;
    DeadDefElimPass pass;
    return pass.run(ir, ctx);
}

}  // namespace

TEST(JitKernelIR, DeadDefChainDiesInOneSweep) {
    IR ir;
    const value_id a = ir.def({}, stub(), "a");
    const value_id b = ir.def({a}, stub(), "b");
    ir.def({b}, stub(), "c");  // nothing reads c, so a, b and c are all dead

    EXPECT_TRUE(run_dce(ir));
    EXPECT_EQ(count_ops_recursive(ir.ops()), 0U) << "the whole chain is dead";
}

TEST(JitKernelIR, KeepsLiveDefsAndDefLessOps) {
    IR ir;
    const value_id live = ir.def({}, stub(), "live");
    ir.use({live}, stub(), "sink");       // def-less: the side-effecting form
    ir.use({}, stub(), "barrier");        // def-less with no reads either

    EXPECT_FALSE(run_dce(ir));
    EXPECT_EQ(count_ops_recursive(ir.ops()), 3U);
}

// Differs from the obvious guess, and matches LLVM: MachineInstr::isDead
// refuses only ordered/volatile accesses, so a plain dead load goes. It
// can only remove a fault, never introduce one.
TEST(JitKernelIR, RemovesDeadLoadButKeepsStore) {
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    record_load(ir, ptr);  // result unread

    ir.use({ptr}, stub(), "store");
    ir.ops().back().may_store = true;

    EXPECT_TRUE(run_dce(ir));
    // The load is gone; the pointer stays because the store reads it, and
    // the store stays because it may write memory.
    EXPECT_EQ(count_ops_recursive(ir.ops()), 2U);
    for (const auto& op : ir.ops()) {
        EXPECT_FALSE(op.may_load) << "dead load survived";
    }
}

TEST(JitKernelIR, DefUsedOnlyInsideLoopBodyIsLive) {
    IR ir;
    const value_id outer = ir.def({}, stub(), "outer");
    const value_id carried = ir.def({}, stub(), "carried");

    ir.loop({carried}, stub(), [&] {
        // Reads a value defined before the loop, and derives a dead value
        // from the loop-carried one.
        ir.use({outer}, stub(), "body_use");
        ir.def({carried}, stub(), "body_def");
    });

    EXPECT_TRUE(run_dce(ir)) << "body_def's result is unread";
    // outer and carried both survive: their uses are inside the region.
    bool saw_outer_def = false;
    bool saw_carried_def = false;
    for (const auto& op : ir.ops()) {
        saw_outer_def |= (op.def == outer);
        saw_carried_def |= (op.def == carried);
    }
    EXPECT_TRUE(saw_outer_def);
    EXPECT_TRUE(saw_carried_def);
}

TEST(JitKernelIR, FoldsSingleUseLoadIntoItsConsumer) {
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    const value_id other = ir.def({}, stub(), "other");
    const value_id loaded = record_load(ir, ptr, /*offset=*/0x40);
    const value_id result = record_consumer(ir, {other, loaded}, 0b10);
    ir.use({result}, stub(), "sink");

    EXPECT_TRUE(run_folding(ir));
    EXPECT_FALSE(has_load(ir.ops())) << "the load should be gone";

    // The consumer now names the base pointer in the folded operand.
    const auto& consumer = *std::next(ir.ops().begin(), 2);
    EXPECT_EQ(consumer.folded_read, 1);
    EXPECT_EQ(consumer.reads[1], ptr);
    EXPECT_EQ(consumer.mem_offset, 0x40U);
    EXPECT_TRUE(consumer.may_load);
}

TEST(JitKernelIR, FoldsCommutedOperandForCommutativeOps) {
    // The loaded value is operand 0. x86 wants the memory operand last, so
    // this only folds when the op says both operands are foldable.
    IR commutative;
    {
        const value_id ptr = commutative.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = commutative.def({}, stub(), "other");
        const value_id loaded = record_load(commutative, ptr);
        const value_id result = record_consumer(commutative, {loaded, other}, 0b11);
        commutative.use({result}, stub(), "sink");
    }
    EXPECT_TRUE(run_folding(commutative));
    EXPECT_FALSE(has_load(commutative.ops()));

    IR non_commutative;
    {
        const value_id ptr = non_commutative.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = non_commutative.def({}, stub(), "other");
        const value_id loaded = record_load(non_commutative, ptr);
        const value_id result = record_consumer(non_commutative, {loaded, other}, 0b10);
        non_commutative.use({result}, stub(), "sink");
    }
    EXPECT_FALSE(run_folding(non_commutative));
    EXPECT_TRUE(has_load(non_commutative.ops()))
        << "operand 0 of a non-commutative op cannot come from memory";
}

TEST(JitKernelIR, FoldsBroadcastOnlyIntoAConsumerThatAcceptsThatForm) {
    // A full-vector operand and a splatted element are different
    // instructions, so a consumer that accepts only the vector form must
    // not take a broadcast: it would read a whole vector where four bytes
    // were meant.
    IR vector_only;
    {
        const value_id ptr = vector_only.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = vector_only.def({}, stub(), "other");
        const value_id splat = record_broadcast(vector_only, ptr);
        const value_id result = record_consumer(vector_only, {other, splat}, 0b10);
        vector_only.use({result}, stub(), "sink");
    }
    EXPECT_FALSE(run_folding(vector_only));
    EXPECT_TRUE(has_load(vector_only.ops()))
        << "a broadcast must not fold into the full-vector form";

    IR accepts_broadcast;
    {
        const value_id ptr = accepts_broadcast.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = accepts_broadcast.def({}, stub(), "other");
        const value_id splat = record_broadcast(accepts_broadcast, ptr, /*offset=*/0x24);
        const value_id result =
            record_broadcast_consumer(accepts_broadcast, {other, splat}, 0b10, 0b10);
        accepts_broadcast.use({result}, stub(), "sink");
    }
    EXPECT_TRUE(run_folding(accepts_broadcast));
    EXPECT_FALSE(has_load(accepts_broadcast.ops()));
    const auto& consumer = *std::next(accepts_broadcast.ops().begin(), 2);
    EXPECT_EQ(consumer.folded_read, 1);
    EXPECT_EQ(consumer.mem_offset, 0x24U);
}

TEST(JitKernelIR, DoesNotFoldAVectorLoadIntoTheBroadcastFormOnly) {
    // The converse: a consumer that accepts only broadcasts must not
    // swallow a full-width load.
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    const value_id other = ir.def({}, stub(), "other");
    const value_id loaded = record_load(ir, ptr);
    const value_id result = record_broadcast_consumer(ir, {other, loaded}, 0b00, 0b10);
    ir.use({result}, stub(), "sink");

    EXPECT_FALSE(run_folding(ir));
    EXPECT_TRUE(has_load(ir.ops()));
}

TEST(JitKernelIR, DoesNotFoldMultiUseBroadcast) {
    // The use count is what makes a GEMM microkernel keep vbroadcastss
    // when several FMAs share the splat, and fold it when only one does.
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    const value_id other = ir.def({}, stub(), "other");
    const value_id splat = record_broadcast(ir, ptr);
    const value_id first = record_broadcast_consumer(ir, {other, splat}, 0b10, 0b10);
    ir.use({splat, first}, stub(), "second_use");

    EXPECT_FALSE(run_folding(ir));
    EXPECT_TRUE(has_load(ir.ops())) << "the splat still has a second consumer";
}

TEST(JitKernelIR, DoesNotFoldMultiUseLoad) {
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    const value_id other = ir.def({}, stub(), "other");
    const value_id loaded = record_load(ir, ptr);
    const value_id first = record_consumer(ir, {other, loaded}, 0b10);
    ir.use({loaded, first}, stub(), "second_use");

    EXPECT_FALSE(run_folding(ir));
    EXPECT_TRUE(has_load(ir.ops())) << "the load still has a second consumer";
}

TEST(JitKernelIR, DoesNotFoldAcrossAStoreOrAPointerBump) {
    // A store between the load and its use may alias the loaded address.
    IR across_store;
    {
        const value_id ptr = across_store.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = across_store.def({}, stub(), "other");
        const value_id loaded = record_load(across_store, ptr);
        across_store.use({other}, stub(), "store");
        across_store.last().may_store = true;
        const value_id result = record_consumer(across_store, {other, loaded}, 0b10);
        across_store.use({result}, stub(), "sink");
    }
    EXPECT_FALSE(run_folding(across_store));
    EXPECT_TRUE(has_load(across_store.ops()));

    // A pointer bump rewrites the register the folded address would use.
    IR across_bump;
    {
        const value_id ptr = across_bump.def({}, stub(), "ptr", RegisterClass::GPR);
        const value_id other = across_bump.def({}, stub(), "other");
        const value_id loaded = record_load(across_bump, ptr);
        across_bump.use({ptr}, stub(), "ptr_advance");
        const value_id result = record_consumer(across_bump, {other, loaded}, 0b10);
        across_bump.use({result}, stub(), "sink");
    }
    EXPECT_FALSE(run_folding(across_bump));
    EXPECT_TRUE(has_load(across_bump.ops()));
}

TEST(JitKernelIR, DoesNotFoldIntoARegionOrPastOne) {
    IR ir;
    const value_id ptr = ir.def({}, stub(), "ptr", RegisterClass::GPR);
    const value_id other = ir.def({}, stub(), "other");
    const value_id loaded = record_load(ir, ptr);
    value_id result = invalid_value;
    ir.region(stub(), [&]() {
        result = record_consumer(ir, {other, loaded}, 0b10);
    });
    ir.use({result}, stub(), "sink");

    EXPECT_FALSE(run_folding(ir));
    EXPECT_TRUE(has_load(ir.ops()))
        << "the consumer is only reached conditionally; the load must stay";
}

// ── Mask register class ────────────────────────────────────────────────

namespace {

std::optional<Assignment> assign_with_masks(IR& ir, std::vector<LiveRange>& ranges,
                                            std::uint32_t vec_pool_size,
                                            const std::vector<std::uint32_t>& gpr_pool,
                                            const std::vector<std::uint32_t>& mask_pool) {
    PassContext ctx;
    for (std::uint32_t i = 0; i < vec_pool_size; ++i) {
        ctx.vec_pool_indices.push_back(i);
    }
    ctx.gpr_pool_indices = gpr_pool;
    ctx.mask_pool_indices = mask_pool;
    return assign_registers(ir, ranges, ctx);
}

}  // namespace

TEST(JitKernelIR, MaskValuesAllocateFromTheirOwnFile) {
    // Predicates are a register class, not a special case: two live at
    // once get two different mask registers, and a mask register index may
    // coincide with a vec or gpr index without interfering.
    IR ir;
    const value_id vec = ir.def({}, stub(), "vec");
    const value_id m0 = ir.def({}, stub(), "mask0", RegisterClass::Mask);
    const value_id m1 = ir.def({}, stub(), "mask1", RegisterClass::Mask);
    ir.use({vec, m0, m1}, stub(), "use_all");

    auto ranges = compute_live_ranges(ir);
    EXPECT_EQ(ranges[m0].rc, RegisterClass::Mask);

    const auto assignment = assign_with_masks(ir, ranges, /*vec*/ 2, {3}, {1, 2, 3});
    ASSERT_TRUE(assignment.has_value());
    EXPECT_NE(assignment->reg.at(m0).idx, assignment->reg.at(m1).idx);
    EXPECT_GE(assignment->reg.at(m0).idx, 1U) << "k0 is not in the allocation order";
}

TEST(JitKernelIR, MaskCopiesCoalesce) {
    IR ir;
    const value_id src = ir.def({}, stub(), "mask_src", RegisterClass::Mask);
    const value_id cpy = ir.copy(src, stub(), "mask_copy", RegisterClass::Mask);
    ir.use({cpy}, stub(), "use_copy");

    auto ranges = compute_live_ranges(ir);
    const auto assignment = assign_with_masks(ir, ranges, 0, {}, {1, 2});
    ASSERT_TRUE(assignment.has_value());
    EXPECT_EQ(assignment->reg.at(src).idx, assignment->reg.at(cpy).idx);
}

TEST(JitKernelIR, MaskAllocationFailsWhenTheIsaHasNoPredicates) {
    // AVX2/SSE/NEON report an empty predicate pool. Asking for a mask must
    // fail loudly rather than pick a register that does not exist.
    IR ir;
    const value_id m = ir.def({}, stub(), "mask", RegisterClass::Mask);
    ir.use({m}, stub(), "use_mask");

    auto ranges = compute_live_ranges(ir);
    EXPECT_THROW(assign_with_masks(ir, ranges, 4, {3}, {}), allocation_failure);
}

// ── Early-clobber defs ─────────────────────────────────────────────────

TEST(JitKernelIR, EarlyClobberDefDoesNotShareARegisterWithItsReads) {
    // A plain def may take the register of a read that dies at the same op
    // — reads happen at the early slot, defs at the late slot, so they do
    // not interfere. That is wrong for multi-instruction expansions that
    // write the destination before consuming the operand (computing an
    // active-lane mask, for instance), which is what an early-clobber def
    // expresses: the def starts at the early slot instead.
    const std::vector<std::uint32_t> gpr_pool = {3, 7};

    IR plain;
    const value_id p_src = plain.def({}, stub(), "src", RegisterClass::GPR);
    const value_id p_def = plain.def({p_src}, stub(), "plain", RegisterClass::GPR);
    plain.use({p_def}, stub(), "use");

    auto plain_ranges = compute_live_ranges(plain);
    const auto plain_alloc = assign_registers_test(plain, plain_ranges, 0, gpr_pool);
    ASSERT_TRUE(plain_alloc.has_value());
    EXPECT_EQ(plain_alloc->reg.at(p_src).idx, plain_alloc->reg.at(p_def).idx)
        << "a plain def should be free to reuse a dying read's register";

    IR early;
    const value_id e_src = early.def({}, stub(), "src", RegisterClass::GPR);
    const value_id e_def =
        early.def_early_clobber({e_src}, stub(), "early", RegisterClass::GPR);
    early.use({e_def}, stub(), "use");

    auto early_ranges = compute_live_ranges(early);
    EXPECT_TRUE(early_ranges[e_src].overlaps(early_ranges[e_def]));

    const auto early_alloc = assign_registers_test(early, early_ranges, 0, gpr_pool);
    ASSERT_TRUE(early_alloc.has_value());
    EXPECT_NE(early_alloc->reg.at(e_src).idx, early_alloc->reg.at(e_def).idx)
        << "an early-clobber def must not land on its own read";

    // With a single register there is nowhere to put it: the allocator
    // either rematerializes the source (rewriting the IR, hence nullopt) or
    // fails — what it must not do is hand out the one register twice.
    IR tight;
    const value_id t_src = tight.def({}, stub(), "src", RegisterClass::GPR);
    (void)tight.def_early_clobber({t_src}, stub(), "early", RegisterClass::GPR);
    auto tight_ranges = compute_live_ranges(tight);
    const auto tight_alloc = assign_registers_test(tight, tight_ranges, 0, {5});
    EXPECT_FALSE(tight_alloc.has_value())
        << "one register cannot satisfy an early-clobber def plus its read";
}

// ── Liveness across region boundaries (dataflow, not tree walk) ────────
//
// These are the cases a per-nesting-level walk gets wrong: the def and the
// use sit at different nesting levels, so a walk produces two disjoint
// segments with a hole between them, and the allocator happily hands the
// register to something defined inside that hole. Block-level dataflow
// keeps the value live along every path from def to use.

TEST(JitKernelIR, LivenessCoversLoopCarriedValueAroundBackEdge) {
    // %acc is defined before the loop, read inside the body, and read again
    // by an in-place redefinition (the post-TwoAddressPass shape of an
    // accumulator or an advancing pointer). The value travels around the
    // back edge, so it must be live from the start of the body, *before*
    // the read — a per-nesting-level walk only covers [read, redef) and
    // leaves the wrap-around uncovered, which lets a value defined at the
    // top of the body steal the register.
    value_id acc = invalid_value;
    value_id idx = invalid_value;
    value_id body_tmp = invalid_value;

    auto build = [&](IR& ir) {
        acc = ir.def({}, stub(), "acc");                            // 0
        idx = ir.def({}, stub(), "idx", RegisterClass::GPR);        // 1
        ir.loop({idx}, stub(), [&]() {                              // 2 (header)
            body_tmp = ir.def({}, stub(), "body_tmp");              // 3
            ir.use({acc, body_tmp}, stub(), "read_acc");            // 4
            (void)ir.def({acc}, stub(), "redef_acc");               // 5
            ir.use({idx}, stub(), "loop_footer");                   // 6 (latch)
        });
    };

    IR ir;
    build(ir);
    auto ranges = compute_live_ranges(ir);

    // Live from its def through the whole body: covers the body entry
    // (early slot of op 3 = 6) even though the first read is at op 4.
    EXPECT_TRUE(ranges[acc].liveAt(6))
        << "%acc must be live at the top of the body — it arrives via the back edge";
    EXPECT_TRUE(ranges[acc].overlaps(ranges[body_tmp]));

    // Therefore they must not share a register.
    const std::vector<std::uint32_t> gpr_pool = {3, 7};
    const auto assignment = assign_registers_test(ir, ranges, /*vec_pool_size=*/3, gpr_pool);
    ASSERT_TRUE(assignment.has_value());
    EXPECT_NE(assignment->reg.at(acc).idx, assignment->reg.at(body_tmp).idx);
}

TEST(JitKernelIR, LivenessKeepsLoopInvariantValueLiveForWholeBody) {
    // %c is loop-invariant: defined before the loop, read once inside. It
    // has to stay live for the entire body because the next iteration
    // reads it again — the read is not the end of its range.
    IR ir;
    const value_id c = ir.def({}, stub(), "c");                         // 0
    const value_id idx = ir.def({}, stub(), "idx", RegisterClass::GPR); // 1

    value_id late_tmp = invalid_value;
    ir.loop({idx}, stub(), [&]() {                  // 2 (header)
        ir.use({c}, stub(), "read_c");              // 3
        late_tmp = ir.def({}, stub(), "late_tmp");  // 4 — after c's last read
        ir.use({late_tmp}, stub(), "use_tmp");      // 5
        ir.use({idx}, stub(), "loop_footer");       // 6 (latch)
    });

    auto ranges = compute_live_ranges(ir);

    // Live past its read, to the end of the body (latch late slot + 1).
    EXPECT_GE(ranges[c].endIndex(), 2U * 6U + 2U);
    EXPECT_TRUE(ranges[c].liveAt(2 * 5));
    EXPECT_TRUE(ranges[c].overlaps(ranges[late_tmp]));
}

TEST(JitKernelIR, LivenessSpansLoopForValueUsedOnlyAfterIt) {
    // %a is defined before the loop and used only after it. It must stay
    // live across the whole loop body — the loop can iterate any number of
    // times, so its register cannot be reused inside the body.
    IR ir;
    const value_id a = ir.def({}, stub());       // index 0
    const value_id idx = ir.def({}, stub(), "idx", RegisterClass::GPR);  // index 1

    value_id body_val = invalid_value;
    ir.loop({idx}, stub(), [&]() {               // index 2 (header)
        body_val = ir.def({}, stub());           // index 3
        ir.use({body_val}, stub());              // index 4
        ir.use({idx}, stub(), "loop_footer");    // index 5 (latch)
    });

    ir.use({a}, stub());                         // index 6

    auto ranges = compute_live_ranges(ir);

    EXPECT_EQ(ranges[a].segments.size(), 1U);
    EXPECT_EQ(ranges[a].beginIndex(), 1U);
    EXPECT_EQ(ranges[a].endIndex(), 13U);        // early(6)+1
    EXPECT_TRUE(ranges[a].liveAt(7)) << "%a must be live inside the loop body";
    EXPECT_TRUE(ranges[a].overlaps(ranges[body_val]));

    // The loop counter is live across the back edge: header read, latch
    // read, and every point in between.
    EXPECT_TRUE(ranges[idx].liveAt(2 * 4));      // early slot of the body use
    EXPECT_GE(ranges[idx].endIndex(), 2U * 5U);
}

TEST(JitKernelIR, LivenessKeepsBranchLocalValuesDisjoint) {
    // The flip side: values that die inside their own region must NOT be
    // extended, otherwise branch-local temporaries stop sharing registers
    // (this is what keeps store_interleaved3's intermediates cheap).
    IR ir;
    value_id then_tmp = invalid_value;
    value_id else_tmp = invalid_value;

    ir.region(stub(), [&]() {
        then_tmp = ir.def({}, stub());
        ir.use({then_tmp}, stub());
    });
    ir.region(stub(), [&]() {
        else_tmp = ir.def({}, stub());
        ir.use({else_tmp}, stub());
    });

    auto ranges = compute_live_ranges(ir);
    EXPECT_FALSE(ranges[then_tmp].overlaps(ranges[else_tmp]));

    const auto assignment = assign_registers_test(ir, ranges, /*pool_size=*/1);
    ASSERT_TRUE(assignment.has_value()) << "branch-local values should share one register";
    EXPECT_EQ(assignment->reg.at(then_tmp).idx, assignment->reg.at(else_tmp).idx);
}

// ── Slice 2: end-to-end integration tests ──────────────────────────────
//
// These tests exercise the full IR-mode path through the real jit_kernel
// DSL: record ops via variable operators, run allocation, lower to xbyak,
// execute the generated kernel, and verify output.

namespace {

struct VecAddParams {
    const float* a;
    const float* b;
    float* result;
};

// Kernel that adds two vectors using IR mode.
// The operator+ goes through vec_op() → IR recording → assignment → lowering.
template <size_t N>
struct jit_ir_vec_add_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_vec_add_kernel)

    jit_ir_vec_add_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const VecAddParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const VecAddParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        begin_ir();

        auto a_ptr = arg(&VecAddParams::a);
        auto b_ptr = arg(&VecAddParams::b);
        auto r_ptr = arg(&VecAddParams::result);

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);
        auto sum = vaddps(a, b);
        ir_store<N>(r_ptr, sum);

        end_ir();

        postamble();
    }
};

}  // namespace

TEST(JitKernelIR, EndToEndVecAdd) {
    using namespace dnnl::impl::cpu::x64;

    // Test with AVX2 (8 floats) if available
    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_vec_add_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(100 + i);
        }
        VecAddParams args{a.data(), b.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i] + b[i])
                << "mismatch at index " << i;
        }
    }
}

// Kernel that computes (a + b) * (a - b) using IR mode with sugar syntax.
// Exercises vaddps, vsubps, vmulps functors and multi-op IR.
template <size_t N>
struct jit_ir_vec_expr_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_vec_expr_kernel)

    jit_ir_vec_expr_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const VecAddParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const VecAddParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        begin_ir();

        auto a_ptr = arg(&VecAddParams::a);
        auto b_ptr = arg(&VecAddParams::b);
        auto r_ptr = arg(&VecAddParams::result);

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);

        // (a + b) * (a - b)
        auto sum  = vaddps(a, b);
        auto diff = vsubps(a, b);
        auto res  = vmulps(sum, diff);

        // Also compute a*a + b*b via fma to verify FMA path
        // (not stored — just exercises the recording path for dead values)
        (void)fma(a, a, vmulps(b, b));

        ir_store<N>(r_ptr, res);

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndVecExpr) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_vec_expr_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i) * 0.5f;
        }
        VecAddParams args{a.data(), b.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], (a[i] + b[i]) * (a[i] - b[i]))
                << "mismatch at index " << i;
        }
    }
}

// Kernel that computes fma(a, b, c) = a*b + c using IR mode.
struct FmaParams {
    const float* a;
    const float* b;
    const float* c;
    float* result;
};

template <size_t N>
struct jit_ir_fma_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_fma_kernel)

    jit_ir_fma_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const FmaParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const FmaParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        begin_ir();

        auto a_ptr = arg(&FmaParams::a);
        auto b_ptr = arg(&FmaParams::b);
        auto c_ptr = arg(&FmaParams::c);
        auto r_ptr = arg(&FmaParams::result);

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);
        auto c = ir_load<N>(c_ptr);

        auto res = fma(a, b, c);

        ir_store<N>(r_ptr, res);

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndFma) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        jit_ir_fma_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, N> a{}, b{}, c{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = 2.0f;
            c[i] = static_cast<float>(i) * 0.5f;
        }
        FmaParams args{a.data(), b.data(), c.data(), result.data()};
        kernel(args);

        for (size_t i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(result[i], a[i] * b[i] + c[i])
                << "fma mismatch at index " << i;
        }
    }
}

// ── Slice 3: foreach in IR mode ────────────────────────────────────────

// Kernel that scales an array by a constant: dst[i] = src[i] * scale.
// The scale is loaded BEFORE the loop (loop-invariant). If the allocator
// doesn't extend its interval across the loop, the scale register gets
// reused and the kernel produces garbage.
struct ScaleParams {
    const float* src;
    float* dst;
    size_t count;    // number of vectors to process
    float scale;
};

template <size_t N>
struct jit_ir_scale_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_scale_kernel)

    jit_ir_scale_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const ScaleParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const ScaleParams& args) const { fn_(&args); }

    void generate() override {
        using reg_type = typename reg_traits<float[N]>::type;
        preamble();

        begin_ir();

        auto src_ptr = arg(&ScaleParams::src);
        auto dst_ptr = arg(&ScaleParams::dst);
        auto count   = arg(&ScaleParams::count);

        // Broadcast scale — loop-invariant, defined before foreach.
        // Tests that the allocator keeps scale live across the loop.
        auto scale_addr = argPtr(&ScaleParams::scale);
        auto scale = ir_def<N>({}, [this, scale_addr](const jit_kernel_ir::EmitContext& ctx) {
            uni_vbroadcastss(reg_type(ctx.def->idx), scale_addr);
        });

        foreach(size_t{0}, count, [&](const variable<size_t>& idx) {
            auto x = ir_load<N>(src_ptr);
            auto result = vmulps(x, scale);
            ir_store<N>(dst_ptr, result);

            ir_advance(src_ptr, N * sizeof(float));
            ir_advance(dst_ptr, N * sizeof(float));
        });

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndForeach) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;
        constexpr size_t num_vectors = 4;
        constexpr size_t total = N * num_vectors;

        jit_ir_scale_kernel<N> kernel;
        kernel.init();

        alignas(32) std::array<float, total> src{}, dst{};
        for (size_t i = 0; i < total; ++i) {
            src[i] = static_cast<float>(i + 1);
        }
        const float scale = 2.5f;
        ScaleParams args{src.data(), dst.data(), num_vectors, scale};
        kernel(args);

        for (size_t i = 0; i < total; ++i) {
            EXPECT_FLOAT_EQ(dst[i], src[i] * scale)
                << "mismatch at index " << i;
        }
    }
}

// ── Slice 3b: _if/_then/_else in IR mode ───────────────────────────────

// Kernel: if (flag == 0) dst = a + b; else dst = a - b;
struct IfElseParams {
    const float* a;
    const float* b;
    float* result;
    size_t flag;
};

struct Interleave3Params {
    const float* a;
    const float* b;
    const float* c;
    float* dst;
};

struct ForeachBranchInterleave3Params {
    const float* y;
    const float* u;
    const float* v;
    float* dst;
    size_t count;
    size_t flag;
    const float* consts;
};

template <size_t N>
struct jit_ir_interleave3_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_interleave3_kernel)

    explicit jit_ir_interleave3_kernel(bool reverse_order)
        : jit_kernel(jit_name()),
          reverse_order(reverse_order) {}

    using fn_t = void (*)(const Interleave3Params*);
    fn_t fn_ = nullptr;
    bool reverse_order = false;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const Interleave3Params& args) const { fn_(&args); }

    void generate() override {
        preamble();

        begin_ir();

        auto a_ptr = arg(&Interleave3Params::a);
        auto b_ptr = arg(&Interleave3Params::b);
        auto c_ptr = arg(&Interleave3Params::c);
        auto dst_ptr = arg(&Interleave3Params::dst);
        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);
        auto c = ir_load<N>(c_ptr);

        if (reverse_order) {
            store_interleaved3(dst_ptr, c, b, a);
        } else {
            store_interleaved3(dst_ptr, a, b, c);
        }

        end_ir();
        postamble();
    }
};

template <size_t N>
struct jit_ir_foreach_branch_interleave3_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_foreach_branch_interleave3_kernel)

    jit_ir_foreach_branch_interleave3_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const ForeachBranchInterleave3Params*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const ForeachBranchInterleave3Params& args) const { fn_(&args); }

    void generate() override {
        using reg_type = typename reg_traits<float[N]>::type;

        preamble();

        begin_ir();

        auto y_ptr = arg(&ForeachBranchInterleave3Params::y);
        auto u_ptr = arg(&ForeachBranchInterleave3Params::u);
        auto v_ptr = arg(&ForeachBranchInterleave3Params::v);
        auto dst_ptr = arg(&ForeachBranchInterleave3Params::dst);
        auto count = arg(&ForeachBranchInterleave3Params::count);
        auto flag = arg(&ForeachBranchInterleave3Params::flag);
        auto consts_ptr = arg(&ForeachBranchInterleave3Params::consts);

        // Broadcasts read through the IR-managed consts pointer.
        auto bc = [&](int slot) {
            return ir_broadcast<N>(consts_ptr, static_cast<size_t>(slot) * sizeof(float));
        };

        auto y_off = bc(0);
        auto uv_off = bc(1);
        auto y_scale = bc(2);
        auto v_to_r = bc(3);
        auto u_to_g = bc(4);
        auto u_to_b = bc(5);
        auto v_to_g = bc(6);
        auto clamp_hi = bc(7);
        auto clamp_lo = ir_def<N>({}, [this](const jit_kernel_ir::EmitContext& ctx) {
            uni_vxorps(reg_type(ctx.def->idx), reg_type(ctx.def->idx), reg_type(ctx.def->idx));
        }, "vxorps");

        foreach(size_t{0}, count, [&](const variable<size_t>&) {
            auto y = ir_load<N>(y_ptr);
            auto u = ir_load<N>(u_ptr);
            auto v = ir_load<N>(v_ptr);

            y = (y - y_off) * y_scale;
            u = u - uv_off;
            v = v - uv_off;

            auto r = fma(v_to_r, v, y);
            auto g = fnma(v_to_g, v, fnma(u_to_g, u, y));
            auto b = fma(u_to_b, u, y);

            r = r.clamp(clamp_lo, clamp_hi);
            g = g.clamp(clamp_lo, clamp_hi);
            b = b.clamp(clamp_lo, clamp_hi);

            ir_cmp(flag, size_t{0});
            ir_if(cond::not_equal,
                  [&]() { store_interleaved3(dst_ptr, r, g, b); },
                  [&]() { store_interleaved3(dst_ptr, b, g, r); });

            ir_advance(y_ptr, N * sizeof(float));
            ir_advance(u_ptr, N * sizeof(float));
            ir_advance(v_ptr, N * sizeof(float));
            ir_advance(dst_ptr, 3 * N * sizeof(float));
        });

        end_ir();
        postamble();
    }
};

template <size_t N>
struct jit_ir_if_else_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_if_else_kernel)

    jit_ir_if_else_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const IfElseParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const IfElseParams& args) const { fn_(&args); }

    void generate() override {
        preamble();

        begin_ir();

        auto a_ptr = arg(&IfElseParams::a);
        auto b_ptr = arg(&IfElseParams::b);
        auto r_ptr = arg(&IfElseParams::result);
        auto flag  = arg(&IfElseParams::flag);

        auto a = ir_load<N>(a_ptr);
        auto b = ir_load<N>(b_ptr);

        // if (flag == 0) result = a + b; else result = a - b;
        ir_cmp(flag, size_t{0});
        ir_if(cond::not_equal,
            [&]() {
                // then: flag == 0 → store a + b
                auto sum = vaddps(a, b);
                ir_store<N>(r_ptr, sum);
            },
            [&]() {
                // else: flag != 0 → store a - b
                auto diff = vsubps(a, b);
                ir_store<N>(r_ptr, diff);
            });

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndIfElse) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx2)) {
        constexpr size_t N = 8;

        alignas(32) std::array<float, N> a{}, b{}, result{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i + 1);
            b[i] = static_cast<float>(i) * 0.5f;
        }

        // Test flag == 0 → a + b
        {
            jit_ir_if_else_kernel<N> kernel;
            kernel.init();
            IfElseParams args{a.data(), b.data(), result.data(), 0};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(result[i], a[i] + b[i])
                    << "then-branch mismatch at index " << i;
            }
        }

        // Test flag != 0 → a - b
        {
            jit_ir_if_else_kernel<N> kernel;
            kernel.init();
            IfElseParams args{a.data(), b.data(), result.data(), 1};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(result[i], a[i] - b[i])
                    << "else-branch mismatch at index " << i;
            }
        }
    }
}

TEST(JitKernelIR, EndToEndStoreInterleaved3) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        constexpr size_t N = 16;

        alignas(64) std::array<float, N> a{}, b{}, c{};
        alignas(64) std::array<float, 3 * N> rgb{}, bgr{};
        for (size_t i = 0; i < N; ++i) {
            a[i] = static_cast<float>(100 + i);
            b[i] = static_cast<float>(200 + i);
            c[i] = static_cast<float>(300 + i);
        }

        {
            jit_ir_interleave3_kernel<N> kernel(/*reverse_order=*/false);
            kernel.init();
            Interleave3Params args{a.data(), b.data(), c.data(), rgb.data()};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(rgb[i * 3 + 0], a[i]) << "RGB channel 0 mismatch at " << i;
                EXPECT_FLOAT_EQ(rgb[i * 3 + 1], b[i]) << "RGB channel 1 mismatch at " << i;
                EXPECT_FLOAT_EQ(rgb[i * 3 + 2], c[i]) << "RGB channel 2 mismatch at " << i;
            }
        }

        {
            jit_ir_interleave3_kernel<N> kernel(/*reverse_order=*/true);
            kernel.init();
            Interleave3Params args{a.data(), b.data(), c.data(), bgr.data()};
            kernel(args);
            for (size_t i = 0; i < N; ++i) {
                EXPECT_FLOAT_EQ(bgr[i * 3 + 0], c[i]) << "BGR channel 0 mismatch at " << i;
                EXPECT_FLOAT_EQ(bgr[i * 3 + 1], b[i]) << "BGR channel 1 mismatch at " << i;
                EXPECT_FLOAT_EQ(bgr[i * 3 + 2], a[i]) << "BGR channel 2 mismatch at " << i;
            }
        }
    }
}

TEST(JitKernelIR, EndToEndForeachIfElseStoreInterleaved3WithSubIntervals) {
    // This kernel exceeds the register pool: 9 constants + 3 clamped
    // values hold registers across both branches, leaving too few for
    // store_interleaved3 intermediates. The interference-based allocator
    // correctly shares registers for branch-local intermediates, but the
    // pre-branch values still dominate. Sibling-branch remat (future)
    // would rematerialize constants inside each branch to free registers.
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        constexpr size_t N = 16;
        // @todo claude: multi-iteration (num_vectors>1) has a pointer-advance
        // bug in the kernel that predates the allocator change — second
        // iteration writes to wrong dst offset. Investigate separately.
        constexpr size_t num_vectors = 1;
        constexpr size_t total = N * num_vectors;

        alignas(64) std::array<float, total> y{}, u{}, v{};
        alignas(64) std::array<float, 3 * total> rgb{}, bgr{};
        alignas(64) std::array<float, 8> consts{{16.0f, 128.0f, 1.164f, 1.596f, 0.391f, 2.018f, 0.813f, 255.0f}};

        for (size_t i = 0; i < total; ++i) {
            y[i] = (i % 3 == 0) ? 8.0f + static_cast<float>(i) : 200.0f + static_cast<float>(i * 3);
            u[i] = 80.0f + static_cast<float>((i * 13) % 96);
            v[i] = 80.0f + static_cast<float>((i * 17) % 96);
        }

        auto expect_triplet = [&](std::array<float, 3 * total>& out, bool rgb_order) {
            auto clip = [](float x) {
                return std::min(std::max(x, 0.0f), 255.0f);
            };
            for (size_t i = 0; i < total; ++i) {
                const float yv = (y[i] - consts[0]) * consts[2];
                const float uv = u[i] - consts[1];
                const float vv = v[i] - consts[1];
                const float r = clip(yv + consts[3] * vv);
                const float g = clip(yv - consts[4] * uv - consts[6] * vv);
                const float bch = clip(yv + consts[5] * uv);

                if (rgb_order) {
                    EXPECT_FLOAT_EQ(out[i * 3 + 0], r) << "RGB r mismatch at " << i;
                    EXPECT_FLOAT_EQ(out[i * 3 + 1], g) << "RGB g mismatch at " << i;
                    EXPECT_FLOAT_EQ(out[i * 3 + 2], bch) << "RGB b mismatch at " << i;
                } else {
                    EXPECT_FLOAT_EQ(out[i * 3 + 0], bch) << "BGR b mismatch at " << i;
                    EXPECT_FLOAT_EQ(out[i * 3 + 1], g) << "BGR g mismatch at " << i;
                    EXPECT_FLOAT_EQ(out[i * 3 + 2], r) << "BGR r mismatch at " << i;
                }
            }
        };

        // With integrated remat, the allocator rematerializes long-lived
        // constants inside each branch, freeing registers for branch-local
        // intermediates. The kernel now fits.
        // flag=0 → jne not taken → builder 1 → store(r,g,b) → RGB order
        {
            jit_ir_foreach_branch_interleave3_kernel<N> kernel;
            EXPECT_NO_THROW(kernel.init());
            ForeachBranchInterleave3Params args{
                y.data(), u.data(), v.data(), rgb.data(),
                num_vectors, 0, consts.data()};
            kernel(args);
            expect_triplet(rgb, true);
        }

        // flag=1 → jne taken → builder 2 → store(b,g,r) → BGR order
        {
            jit_ir_foreach_branch_interleave3_kernel<N> kernel;
            EXPECT_NO_THROW(kernel.init());
            ForeachBranchInterleave3Params args{
                y.data(), u.data(), v.data(), bgr.data(),
                num_vectors, 1, consts.data()};
            kernel(args);
            expect_triplet(bgr, false);
        }
    }
}

// ── foreach_predicated: masked load/store ──────────────────────────────

// Kernel: dst[i] = src[i] * 2.0f, using predicated foreach.
// Tests that the tail (count % N) is handled by the mask, producing
// correct results without a separate eager-mode tail path.
struct PredicatedScaleParams {
    const float* src;
    float* dst;
    size_t count;   // total elements (may not be a multiple of N)
    float scale;
};

template <size_t N>
struct jit_ir_predicated_scale_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_predicated_scale_kernel)

    jit_ir_predicated_scale_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const PredicatedScaleParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success)
            OPENVINO_THROW("Can't generate jit kernel");
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const PredicatedScaleParams& args) const { fn_(&args); }

    void generate() override {
        using reg_type = typename reg_traits<float[N]>::type;
        preamble();

        auto scale_addr = argPtr(&PredicatedScaleParams::scale);

        begin_ir();

        auto src_ptr = arg(&PredicatedScaleParams::src);
        auto dst_ptr = arg(&PredicatedScaleParams::dst);
        auto count   = arg(&PredicatedScaleParams::count);

        auto scale = ir_def<N>({}, [this, scale_addr](const jit_kernel_ir::EmitContext& ctx) {
            uni_vbroadcastss(reg_type(ctx.def->idx), scale_addr);
        });

        foreach_predicated<N>(count, [&](const vlen& vl) {
            auto x = ir_load<N>(src_ptr, vl);     // masked load
            auto result = vmulps(x, scale);
            ir_store<N>(dst_ptr, result, vl);     // masked store

            // Advance pointers.
            ir_advance(src_ptr, N * sizeof(float));
            ir_advance(dst_ptr, N * sizeof(float));
        });

        end_ir();
        postamble();
    }
};

TEST(JitKernelIR, EndToEndForeachPredicated) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        constexpr size_t N = 16;

        // Test with count that is NOT a multiple of N.
        // 25 elements = 1 full iteration (16) + 1 tail iteration (9).
        constexpr size_t count = 25;

        alignas(64) std::array<float, count + N> src{};  // pad to avoid overread
        alignas(64) std::array<float, count + N> dst{};
        for (size_t i = 0; i < count; ++i) {
            src[i] = static_cast<float>(i + 1);
        }
        // Fill padding with sentinel to detect overwrite.
        for (size_t i = count; i < count + N; ++i) {
            src[i] = -999.0f;
            dst[i] = -999.0f;
        }

        jit_ir_predicated_scale_kernel<N> kernel;
        kernel.init();

        PredicatedScaleParams args{src.data(), dst.data(), count, 2.0f};
        kernel(args);

        for (size_t i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(dst[i], src[i] * 2.0f)
                << "mismatch at index " << i;
        }

        // Verify no overwrite past count.
        for (size_t i = count; i < count + N; ++i) {
            EXPECT_FLOAT_EQ(dst[i], -999.0f)
                << "masked store overwrote past count at index " << i;
        }
    }
}

// ── GPR register class tests ─────────────────────────────────────────

TEST(JitKernelIR, GprAllocation) {
    // GPR-only IR: two GPR defs, allocated from GPR pool.
    jit_kernel_ir::IR ir;
    auto v0 = ir.def({}, [](const auto&){}, "gpr_a", jit_kernel_ir::RegisterClass::GPR);
    auto v1 = ir.def({}, [](const auto&){}, "gpr_b", jit_kernel_ir::RegisterClass::GPR);
    ir.use({v0, v1}, [](const auto&){}, "use_both");

    auto ranges = jit_kernel_ir::compute_live_ranges(ir);
    ASSERT_EQ(ranges[v0].rc, jit_kernel_ir::RegisterClass::GPR);
    ASSERT_EQ(ranges[v1].rc, jit_kernel_ir::RegisterClass::GPR);

    // Allocate with no vec pool, 4-entry GPR pool (indices 3, 6, 8, 10).
    std::vector<std::uint32_t> gpr_pool = {3, 6, 8, 10};
    auto result = assign_registers_test(ir, ranges, /*vec_pool_size=*/0, gpr_pool);
    ASSERT_TRUE(result.has_value());

    // Both values should be assigned physical GPR indices from the pool.
    auto p0 = result->reg.at(v0).idx;
    auto p1 = result->reg.at(v1).idx;
    EXPECT_NE(p0, p1);
    // Both must be in the GPR pool.
    auto in_pool = [&](std::uint32_t idx) {
        return std::find(gpr_pool.begin(), gpr_pool.end(), idx) != gpr_pool.end();
    };
    EXPECT_TRUE(in_pool(p0)) << "p0=" << p0 << " not in GPR pool";
    EXPECT_TRUE(in_pool(p1)) << "p1=" << p1 << " not in GPR pool";
}

TEST(JitKernelIR, MixedVecGprAllocation) {
    // Mix of Vec and GPR values — separate pools, no cross-class interference.
    jit_kernel_ir::IR ir;
    auto vec0 = ir.def({}, [](const auto&){}, "vec0");  // default = Vec
    auto gpr0 = ir.def({}, [](const auto&){}, "gpr0", jit_kernel_ir::RegisterClass::GPR);
    auto vec1 = ir.def({vec0}, [](const auto&){}, "vec1");
    auto gpr1 = ir.def({gpr0}, [](const auto&){}, "gpr1", jit_kernel_ir::RegisterClass::GPR);
    ir.use({vec0, vec1, gpr0, gpr1}, [](const auto&){}, "use_all");

    auto ranges = jit_kernel_ir::compute_live_ranges(ir);
    EXPECT_EQ(ranges[vec0].rc, jit_kernel_ir::RegisterClass::Vec);
    EXPECT_EQ(ranges[gpr0].rc, jit_kernel_ir::RegisterClass::GPR);

    // Vec pool size = 2, GPR pool = {5, 9}
    std::vector<std::uint32_t> gpr_pool = {5, 9};
    auto result = assign_registers_test(ir, ranges, /*vec_pool_size=*/2, gpr_pool);
    ASSERT_TRUE(result.has_value());

    // Vec values get indices 0..1, GPR values get indices from {5, 9}.
    auto pv0 = result->reg.at(vec0).idx;
    auto pv1 = result->reg.at(vec1).idx;
    auto pg0 = result->reg.at(gpr0).idx;
    auto pg1 = result->reg.at(gpr1).idx;

    EXPECT_LT(pv0, 2u);
    EXPECT_LT(pv1, 2u);
    EXPECT_TRUE(pg0 == 5 || pg0 == 9) << "pg0=" << pg0;
    EXPECT_TRUE(pg1 == 5 || pg1 == 9) << "pg1=" << pg1;

    // Vec idx 0 and GPR idx 0 can coexist — different register files.
    // No cross-class conflict even if indices overlap.
}

TEST(JitKernelIR, GprCoalescing) {
    // GPR copy should coalesce to the same physical register.
    jit_kernel_ir::IR ir;
    auto src = ir.def({}, [](const auto&){}, "gpr_src", jit_kernel_ir::RegisterClass::GPR);
    auto cpy = ir.copy(src, [](const auto&){}, "gpr_copy", jit_kernel_ir::RegisterClass::GPR);
    ir.use({cpy}, [](const auto&){}, "use_copy");

    auto ranges = jit_kernel_ir::compute_live_ranges(ir);
    std::vector<std::uint32_t> gpr_pool = {3, 7};
    auto result = assign_registers_test(ir, ranges, /*vec_pool_size=*/0, gpr_pool);
    ASSERT_TRUE(result.has_value());

    // Copy should be coalesced — same physical register.
    EXPECT_EQ(result->reg.at(src).idx, result->reg.at(cpy).idx);
}

TEST(JitKernelIR, GprPoolOverflow) {
    // GPR pool of 1 with 2 simultaneously live GPR values.
    // Vec pool has plenty of room — only GPR is constrained.
    // With remat, the allocator may return nullopt (IR modified) instead of
    // throwing. Either outcome indicates the GPR pool is too small for a
    // single-pass allocation.
    jit_kernel_ir::IR ir;
    auto v0 = ir.def({}, [](const auto&){}, "gpr0", jit_kernel_ir::RegisterClass::GPR);
    auto v1 = ir.def({v0}, [](const auto&){}, "gpr1", jit_kernel_ir::RegisterClass::GPR);
    ir.use({v0, v1}, [](const auto&){}, "use_both");

    auto ranges = jit_kernel_ir::compute_live_ranges(ir);
    std::vector<std::uint32_t> gpr_pool = {8};  // only 1 register
    auto result = assign_registers_test(ir, ranges, /*vec_pool_size=*/4, gpr_pool);
    // Remat modifies the IR and returns nullopt — the pool was too small.
    EXPECT_FALSE(result.has_value()) << "expected remat (nullopt), not a successful assignment";
}

// ── Verifier ───────────────────────────────────────────────────────────

TEST(JitKernelIR, VerifierRejectsInterferenceViolation) {
    // Two simultaneously live values forced onto the same register.
    IR ir;
    const value_id a = ir.def({}, stub(), "a");
    const value_id b = ir.def({}, stub(), "b");
    ir.use({a, b}, stub(), "use_both");

    auto ranges = compute_live_ranges(ir);
    PassContext ctx;
    for (std::uint32_t i = 0; i < 2; ++i) ctx.vec_pool_indices.push_back(i);

    Assignment bad;
    bad.reg[a] = PhysReg{0};
    bad.reg[b] = PhysReg{0};
    EXPECT_THROW(verify(ir, ranges, bad, ctx), verification_failure);

    Assignment good;
    good.reg[a] = PhysReg{0};
    good.reg[b] = PhysReg{1};
    EXPECT_NO_THROW(verify(ir, ranges, good, ctx));
}

TEST(JitKernelIR, VerifierRejectsMissingAssignmentAndDanglingRead) {
    IR ir;
    const value_id a = ir.def({}, stub(), "a");
    ir.use({a}, stub(), "use_a");

    auto ranges = compute_live_ranges(ir);
    PassContext ctx;
    for (std::uint32_t i = 0; i < 2; ++i) ctx.vec_pool_indices.push_back(i);

    // Referenced but unassigned — lowering would throw out_of_range instead.
    EXPECT_THROW(verify(ir, ranges, Assignment{}, ctx), verification_failure);

    // Read of a value nothing defines — what a botched IR rewrite produces.
    IR dangling;
    dangling.use({7}, stub(), "read_of_nothing");
    dangling.set_value_count(8);
    auto dangling_ranges = compute_live_ranges(dangling);
    Assignment any;
    any.reg[7] = PhysReg{0};
    EXPECT_THROW(verify(dangling, dangling_ranges, any, ctx), verification_failure);
}

TEST(JitKernelIR, VerifierRejectsOutOfPoolRegister) {
    IR ir;
    const value_id v = ir.def({}, stub(), "v");
    ir.use({v}, stub(), "use_v");
    auto ranges = compute_live_ranges(ir);

    PassContext ctx;
    for (std::uint32_t i = 0; i < 2; ++i) ctx.vec_pool_indices.push_back(i);
    Assignment bad;
    bad.reg[v] = PhysReg{5};  // outside the 2-register pool
    EXPECT_THROW(verify(ir, ranges, bad, ctx), verification_failure);
}

// ── Randomized allocation ──────────────────────────────────────────────

namespace {

// Generates well-formed random IR: values defined inside a region stay
// inside it (so reads never escape their defining scope), mixed register
// classes, tied ops, copies, and nested branch/loop regions.
void build_random_ops(IR& ir,
                      std::mt19937& rng,
                      std::vector<value_id>& vec_vals,
                      std::vector<value_id>& gpr_vals,
                      int op_count,
                      int depth) {
    auto roll = [&rng](int n) { return static_cast<int>(rng() % static_cast<unsigned>(n)); };

    auto pick_reads = [&](const std::vector<value_id>& pool, int max_reads) {
        std::vector<value_id> reads;
        if (pool.empty()) {
            return reads;
        }
        const int count = roll(max_reads + 1);
        for (int k = 0; k < count; ++k) {
            reads.push_back(pool[static_cast<std::size_t>(roll(static_cast<int>(pool.size())))]);
        }
        return reads;
    };

    for (int i = 0; i < op_count; ++i) {
        const int kind = roll(100);
        if (kind < 35) {
            vec_vals.push_back(ir.def(pick_reads(vec_vals, 2), stub(), "rnd_vec"));
        } else if (kind < 55) {
            gpr_vals.push_back(ir.def(pick_reads(gpr_vals, 2), stub(), "rnd_gpr",
                                      RegisterClass::GPR));
        } else if (kind < 65 && !vec_vals.empty()) {
            // FMA-shaped: three reads, first one tied to the def.
            std::vector<value_id> reads = pick_reads(vec_vals, 3);
            while (reads.size() < 3) {
                reads.push_back(vec_vals[static_cast<std::size_t>(roll(static_cast<int>(vec_vals.size())))]);
            }
            vec_vals.push_back(ir.def_tied(reads, /*tied_to=*/0, stub(), "rnd_fma"));
        } else if (kind < 72 && !vec_vals.empty()) {
            vec_vals.push_back(ir.copy(
                vec_vals[static_cast<std::size_t>(roll(static_cast<int>(vec_vals.size())))],
                stub(), "rnd_copy"));
        } else if (kind < 90 || depth >= 2) {
            auto reads = pick_reads(vec_vals, 3);
            if (!gpr_vals.empty()) {
                reads.push_back(gpr_vals[static_cast<std::size_t>(roll(static_cast<int>(gpr_vals.size())))]);
            }
            if (!reads.empty()) {
                ir.use(reads, stub(), "rnd_use");
            }
        } else {
            const bool is_loop = roll(2) == 0;
            std::vector<value_id> header_reads;
            if (!gpr_vals.empty()) {
                header_reads.push_back(
                    gpr_vals[static_cast<std::size_t>(roll(static_cast<int>(gpr_vals.size())))]);
            }
            // Snapshot pool sizes: values defined in the body do not escape.
            const auto vec_mark = vec_vals.size();
            const auto gpr_mark = gpr_vals.size();
            ir.region(header_reads, stub(), [&]() {
                build_random_ops(ir, rng, vec_vals, gpr_vals, 1 + roll(4), depth + 1);
            }, is_loop);
            vec_vals.resize(vec_mark);
            gpr_vals.resize(gpr_mark);
        }
    }
}

}  // namespace

TEST(JitKernelIR, RandomizedAllocationSatisfiesVerifier) {
    // Fuzz the allocator: any successful allocation must satisfy the
    // verifier. Pool exhaustion is a legitimate outcome (no spiller), a
    // verification failure never is.
    std::size_t allocated = 0;
    std::size_t exhausted = 0;

    for (unsigned seed = 0; seed < 400; ++seed) {
        std::mt19937 rng(seed);
        IR ir;
        std::vector<value_id> vec_vals;
        std::vector<value_id> gpr_vals;
        build_random_ops(ir, rng, vec_vals, gpr_vals, 10 + static_cast<int>(seed % 20), 0);

        PassContext ctx;
        for (std::uint32_t i = 0; i < 3 + seed % 14; ++i) ctx.vec_pool_indices.push_back(i);
        for (std::uint32_t g = 0; g < 2 + seed % 7; ++g) {
            ctx.gpr_pool_indices.push_back(g);
        }

        PassManager pm;
        pm.add<TwoAddressPass>();
        pm.add<LiveRangeAnalysis>();
        pm.add<RegisterAllocator>();
        pm.add<VerifyPass>();

        try {
            pm.run(ir, ctx);
        } catch (const allocation_failure&) {
            ++exhausted;  // pool too small for this random program
            continue;
        } catch (const ov::Exception&) {
            ++exhausted;  // remat retries exhausted
            continue;
        }
        ASSERT_TRUE(ctx.assignment.has_value()) << "seed " << seed;
        ++allocated;
    }

    // Sanity: the fuzzer must actually be allocating, not just failing.
    EXPECT_GT(allocated, 100U) << "allocated=" << allocated << " exhausted=" << exhausted;
}

// ── Differential: epilogue kernel vs scalar reference ──────────────────
//
// dst[i] = a[i] * b[i] + c[i] for i < width, driven by
// foreach_with_epilogue. Exercises the main loop, the partial load path
// and the partial store path over widths that are deliberately not
// multiples of the vector width. Sentinels around the destination catch
// stores past `width`.

namespace {

struct FmaEpilogueParams {
    const float* a;
    const float* b;
    const float* c;
    float* dst;
    size_t width;
};

template <size_t N>
struct jit_ir_fma_epilogue_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_fma_epilogue_kernel)

    jit_ir_fma_epilogue_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const FmaEpilogueParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const FmaEpilogueParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        begin_ir();

        auto width = arg(&FmaEpilogueParams::width);
        auto a = make_ir_ptr(arg<const float*>(&FmaEpilogueParams::a), N);
        auto b = make_ir_ptr(arg<const float*>(&FmaEpilogueParams::b), N);
        auto c = make_ir_ptr(arg<const float*>(&FmaEpilogueParams::c), N);
        auto dst = make_ir_ptr(arg<float*>(&FmaEpilogueParams::dst), N);

        foreach_vec<N>(width, [&](const vlen& vl) {
            auto va = ir_load<N>(a, vl);
            auto vb = ir_load<N>(b, vl);
            auto vc = ir_load<N>(c, vl);
            ir_store<N>(dst, size_t{0}, fma(va, vb, vc), vl);

            ir_advance(a, vl);
            ir_advance(b, vl);
            ir_advance(c, vl);
            ir_advance(dst, vl);
        });

        end_ir();
        postamble();
    }
};

template <size_t N>
void run_fma_epilogue_differential() {
    constexpr float sentinel = -123456.0f;
    constexpr size_t pad = 2 * N;

    jit_ir_fma_epilogue_kernel<N> kernel;
    kernel.init();

    std::vector<size_t> widths = {0, 1, 2, 3, N - 1, N, N + 1, 2 * N - 1,
                                  2 * N, 2 * N + 3, 5 * N + 7};
    std::mt19937 rng(20260903);
    std::uniform_int_distribution<size_t> width_dist(1, 6 * N);
    for (int extra = 0; extra < 16; ++extra) {
        widths.push_back(width_dist(rng));
    }

    std::uniform_real_distribution<float> val_dist(-4.0f, 4.0f);

    for (auto width : widths) {
        std::vector<float> a(width + pad);
        std::vector<float> b(width + pad);
        std::vector<float> c(width + pad);
        std::vector<float> dst(width + pad, sentinel);
        for (size_t i = 0; i < width + pad; ++i) {
            a[i] = val_dist(rng);
            b[i] = val_dist(rng);
            c[i] = val_dist(rng);
        }

        FmaEpilogueParams args{a.data(), b.data(), c.data(), dst.data(), width};
        kernel(args);

        for (size_t i = 0; i < width; ++i) {
            EXPECT_FLOAT_EQ(dst[i], std::fma(a[i], b[i], c[i]))
                << "width=" << width << " index=" << i;
        }
        for (size_t i = width; i < width + pad; ++i) {
            EXPECT_FLOAT_EQ(dst[i], sentinel)
                << "store past width=" << width << " at index=" << i;
        }
    }
}

}  // namespace

TEST(JitKernelIR, DifferentialFmaWithEpilogue) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        run_fma_epilogue_differential<16>();
    } else if (mayiuse(cpu_isa_t::avx2)) {
        run_fma_epilogue_differential<8>();
    } else {
        GTEST_SKIP() << "requires AVX2 or AVX-512";
    }
}

// ── Differential: peeled constant trip count vs scalar reference ───────
//
// Same kernel as above, but the trip count is a C++ value, so foreach_vec
// emits the full iterations straight-line and the remainder once. Every
// count is run twice — peeled and (with peel_limit 0) rolled — against the
// same reference, so the two shapes are checked to agree rather than each
// being checked in isolation. Sentinels catch a peeled step storing past
// the count, which is the failure mode of getting the remainder wrong.

namespace {

struct FmaPeelParams {
    const float* a;
    const float* b;
    const float* c;
    float* dst;
};

template <size_t N>
struct jit_ir_fma_peel_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_fma_peel_kernel)

    jit_ir_fma_peel_kernel(size_t count, size_t limit)
        : jit_kernel(jit_name()), _count(count) {
        set_peel_limit(limit);
    }

    using fn_t = void (*)(const FmaPeelParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const FmaPeelParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        begin_ir();

        auto a = make_ir_ptr(arg<const float*>(&FmaPeelParams::a), N);
        auto b = make_ir_ptr(arg<const float*>(&FmaPeelParams::b), N);
        auto c = make_ir_ptr(arg<const float*>(&FmaPeelParams::c), N);
        auto dst = make_ir_ptr(arg<float*>(&FmaPeelParams::dst), N);

        foreach_vec<N>(_count, [&](const vlen& vl) {
            auto va = ir_load<N>(a, vl);
            auto vb = ir_load<N>(b, vl);
            auto vc = ir_load<N>(c, vl);
            ir_store<N>(dst, size_t{0}, fma(va, vb, vc), vl);

            ir_advance(a, vl);
            ir_advance(b, vl);
            ir_advance(c, vl);
            ir_advance(dst, vl);
        });

        end_ir();
        postamble();
    }

private:
    size_t _count;
};

template <size_t N>
void run_fma_peel_differential() {
    constexpr float sentinel = -123456.0f;
    constexpr size_t pad = 2 * N;

    std::vector<size_t> counts = {1, 2, 3, N - 1, N, N + 1, 2 * N, 2 * N + 3,
                                  3 * N, 4 * N, 4 * N + 1, 4 * N + N - 1};
    std::mt19937 rng(20260915);
    std::uniform_int_distribution<size_t> count_dist(1, 6 * N);
    for (int extra = 0; extra < 8; ++extra) {
        counts.push_back(count_dist(rng));
    }

    std::uniform_real_distribution<float> val_dist(-4.0f, 4.0f);

    for (auto count : counts) {
        // peel_limit 0 forces the rolled loop; 64 peels every count here.
        for (size_t limit : {size_t{0}, size_t{64}}) {
            jit_ir_fma_peel_kernel<N> kernel(count, limit);
            kernel.init();

            std::vector<float> a(count + pad);
            std::vector<float> b(count + pad);
            std::vector<float> c(count + pad);
            std::vector<float> dst(count + pad, sentinel);
            for (size_t i = 0; i < count + pad; ++i) {
                a[i] = val_dist(rng);
                b[i] = val_dist(rng);
                c[i] = val_dist(rng);
            }

            FmaPeelParams args{a.data(), b.data(), c.data(), dst.data()};
            kernel(args);

            for (size_t i = 0; i < count; ++i) {
                EXPECT_FLOAT_EQ(dst[i], std::fma(a[i], b[i], c[i]))
                    << "count=" << count << " limit=" << limit << " index=" << i;
            }
            for (size_t i = count; i < count + pad; ++i) {
                EXPECT_FLOAT_EQ(dst[i], sentinel)
                    << "store past count=" << count << " limit=" << limit
                    << " at index=" << i;
            }
        }
    }
}

}  // namespace

TEST(JitKernelIR, DifferentialFmaPeeledConstantCount) {
    using namespace dnnl::impl::cpu::x64;

    if (mayiuse(cpu_isa_t::avx512_core)) {
        run_fma_peel_differential<16>();
    } else if (mayiuse(cpu_isa_t::avx2)) {
        run_fma_peel_differential<8>();
    } else {
        GTEST_SKIP() << "requires AVX2 or AVX-512";
    }
}

// Shape check, not a correctness check: peeling must emit one body per
// full iteration, straight-line. Counts that are exact multiples of N have
// no remainder, so each step up adds exactly one body and the size
// increments must be equal. A silent fallback to the rolled loop would
// flatten them.
//
// Deliberately not "peeled is bigger than rolled": under the epilogue
// style the rolled form records a scalarized partial tail that dwarfs four
// peeled bodies, so that comparison says nothing.
TEST(JitKernelIR, PeelingEmitsOneBodyPerIteration) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core) && !mayiuse(cpu_isa_t::avx2)) {
        GTEST_SKIP() << "requires AVX2 or AVX-512";
    }
    constexpr size_t N = 16;

    std::array<size_t, 4> sizes{};
    for (size_t full = 1; full <= sizes.size(); ++full) {
        jit_ir_fma_peel_kernel<N> kernel(full * N, /*limit=*/8);
        kernel.init();
        sizes[full - 1] = kernel.getSize();
    }

    const size_t body = sizes[1] - sizes[0];
    EXPECT_GT(body, 0U);
    EXPECT_EQ(sizes[2] - sizes[1], body) << "sizes: " << sizes[0] << " " << sizes[1]
                                         << " " << sizes[2] << " " << sizes[3];
    EXPECT_EQ(sizes[3] - sizes[2], body) << "sizes: " << sizes[0] << " " << sizes[1]
                                         << " " << sizes[2] << " " << sizes[3];
}

// ── Decisions taken for a target the host is not ───────────────────────
//
// Both branches below are unreachable on any x86 target, so until the
// target became injectable they were asserted and never executed. RVV is
// the real case: "RVV instructions only support register addressing", so a
// vector access cannot carry a displacement and a peeled loop has to keep
// incrementing its pointers. The same kernel is built twice, once per
// answer, and both must compute the same thing.

namespace {

// Test doubles differ from the host target in one answer each, so they
// share the host's answers for everything else. Deliberately test-only:
// a real target must still implement every query, so that adding one
// forces each architecture to decide rather than inherit a default that
// happens to suit x86.
struct host_like_target : vector_target {
    [[nodiscard]] bool supports_masked_access(std::size_t elem_bytes) const override {
        return elem_bytes == 1 || elem_bytes == 2 || elem_bytes == 4;
    }
    [[nodiscard]] bool supports_masked_interleaved_access() const override { return true; }
    [[nodiscard]] tail_folding preferred_tail_folding() const override {
        return tail_folding::mask;
    }
    [[nodiscard]] bool supports_broadcast_memory_operand(std::size_t elem_bytes) const override {
        return elem_bytes == 4 || elem_bytes == 8;
    }
    [[nodiscard]] bool is_legal_access_offset(std::size_t, std::size_t,
                                              std::size_t bytes) const override {
        return bytes <= 0x7fffffff;
    }
    [[nodiscard]] std::size_t preferred_loop_alignment() const override { return 16; }
    [[nodiscard]] std::size_t max_bytes_for_alignment() const override { return 0; }
    [[nodiscard]] std::size_t cache_line_size() const override { return 64; }
    [[nodiscard]] std::size_t prefetch_distance() const override { return 64; }
    [[nodiscard]] const std::vector<std::uint32_t>& predicate_pool() const override {
        static const std::vector<std::uint32_t> pool{1, 2, 3, 4, 5, 6, 7};
        return pool;
    }
};

// A target that refuses displacements outright — RVV's answer for vector
// accesses.
struct no_offset_target final : host_like_target {
    [[nodiscard]] bool is_legal_access_offset(std::size_t, std::size_t,
                                              std::size_t) const override {
        return false;
    }
};

// Same kernel as the peel differential, with an injected target.
template <size_t N>
struct jit_ir_peel_target_kernel : public jit_ir_fma_peel_kernel<N> {
    jit_ir_peel_target_kernel(size_t count, const vector_target& t)
        : jit_ir_fma_peel_kernel<N>(count, /*limit=*/8) {
        this->set_target(t);
    }
};

// A target without AVX-512's embedded broadcast — AVX2's answer, where a
// splat is always vbroadcastss into a register.
struct no_broadcast_operand_target final : host_like_target {
    [[nodiscard]] bool supports_broadcast_memory_operand(std::size_t) const override {
        return false;
    }
};

}  // namespace

TEST(JitKernelIR, TargetWithoutLegalOffsetKeepsPointerIncrements) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "the injected target mirrors an AVX-512 host";
    }
    constexpr size_t N = 16;
    constexpr size_t count = 4 * N;  // four peeled iterations, no remainder

    const no_offset_target no_offset;
    jit_ir_peel_target_kernel<N> incremented(count, no_offset);
    incremented.init();

    // The host target folds the same displacements away, so the kernel that
    // cannot must be larger — three pointer bumps per iteration boundary.
    jit_ir_fma_peel_kernel<N> displaced(count, /*limit=*/8);
    displaced.init();

    EXPECT_GT(incremented.getSize(), displaced.getSize())
        << "a target that refuses displacements must emit the increments instead";

    // And both must compute the same thing.
    constexpr float sentinel = -321.0f;
    std::mt19937 rng(20260916);
    std::uniform_real_distribution<float> val_dist(-4.0f, 4.0f);

    std::vector<float> a(count + N);
    std::vector<float> b(count + N);
    std::vector<float> c(count + N);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = val_dist(rng);
        b[i] = val_dist(rng);
        c[i] = val_dist(rng);
    }

    std::vector<float> dst_inc(count + N, sentinel);
    std::vector<float> dst_disp(count + N, sentinel);
    FmaPeelParams args_inc{a.data(), b.data(), c.data(), dst_inc.data()};
    FmaPeelParams args_disp{a.data(), b.data(), c.data(), dst_disp.data()};
    incremented(args_inc);
    displaced(args_disp);

    for (size_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(dst_inc[i], std::fma(a[i], b[i], c[i])) << "index " << i;
        EXPECT_FLOAT_EQ(dst_inc[i], dst_disp[i]) << "shapes disagree at index " << i;
    }
    for (size_t i = count; i < count + N; ++i) {
        EXPECT_FLOAT_EQ(dst_inc[i], sentinel) << "stored past the count at " << i;
    }
}

// ── Predicated vs scalarized interleaved store ─────────────────────────
//
// store_interleaved3 under a short active length has two realizations and
// the target picks: predicate the three stores that write the interleave
// out, or build the interleave in a stack slot and copy count*3 elements.
// Both are exercised here in one process by injecting the target, which is
// the only way the losing one stays tested once a target stops choosing
// it.

namespace {

struct InterleaveCountParams {
    const float* a;
    const float* b;
    const float* c;
    float* dst;
    size_t count;
};

// Mirrors the host except for the one answer under test.
template <bool MaskedInterleave>
struct interleave_target final : host_like_target {
    [[nodiscard]] bool supports_masked_interleaved_access() const override {
        return MaskedInterleave;
    }
};

template <size_t N>
struct jit_ir_interleave_count_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_interleave_count_kernel)

    explicit jit_ir_interleave_count_kernel(const vector_target& t) : jit_kernel(jit_name()) {
        set_target(t);
    }

    using fn_t = void (*)(const InterleaveCountParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const InterleaveCountParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        set_vec_width(N * sizeof(float) * 8);
        begin_ir();

        auto a = make_ir_ptr(arg<const float*>(&InterleaveCountParams::a), N);
        auto b = make_ir_ptr(arg<const float*>(&InterleaveCountParams::b), N);
        auto c = make_ir_ptr(arg<const float*>(&InterleaveCountParams::c), N);
        auto dst = make_ir_ptr(arg<float*>(&InterleaveCountParams::dst), 3 * N);
        auto count = arg(&InterleaveCountParams::count);

        // One short iteration, straight to the tail realization.
        store_interleaved3(dst, ir_load<N>(a), ir_load<N>(b), ir_load<N>(c),
                           vlen::elements(count.vid(), /*terminal=*/true));

        end_ir();
        postamble();
    }
};

template <size_t N>
void run_interleave_count(size_t count, std::vector<float>& dst, size_t& code_size,
                          const vector_target& t) {
    constexpr float sentinel = -999.0f;
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i) + 0.5f;
        b[i] = static_cast<float>(i) + 100.5f;
        c[i] = static_cast<float>(i) + 200.5f;
    }
    dst.assign(3 * N + 8, sentinel);

    jit_ir_interleave_count_kernel<N> kernel(t);
    kernel.init();
    code_size = kernel.getSize();

    InterleaveCountParams args{a.data(), b.data(), c.data(), dst.data(), count};
    kernel(args);
}

}  // namespace

TEST(JitKernelIR, PredicatedInterleavedStoreMatchesScalarizedForm) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "the injected targets mirror an AVX-512 host";
    }
    constexpr size_t N = 16;
    constexpr float sentinel = -999.0f;

    const interleave_target<true> predicated;
    const interleave_target<false> scalarized;

    // Counts either side of the per-store boundaries: 3*count crosses into
    // the second output vector at count 6 and the third at count 11, so the
    // mask slices are what these exercise.
    for (size_t count : {size_t{0}, size_t{1}, size_t{5}, size_t{6}, size_t{7},
                         size_t{10}, size_t{11}, size_t{12}, size_t{15}, size_t{16}}) {
        std::vector<float> dst_pred;
        std::vector<float> dst_scal;
        size_t size_pred = 0;
        size_t size_scal = 0;
        run_interleave_count<N>(count, dst_pred, size_pred, predicated);
        run_interleave_count<N>(count, dst_scal, size_scal, scalarized);

        // Reference: dst[3i+0..2] = a[i], b[i], c[i] for i < count.
        for (size_t i = 0; i < count; ++i) {
            EXPECT_FLOAT_EQ(dst_pred[3 * i + 0], static_cast<float>(i) + 0.5f)
                << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(dst_pred[3 * i + 1], static_cast<float>(i) + 100.5f)
                << "count=" << count << " i=" << i;
            EXPECT_FLOAT_EQ(dst_pred[3 * i + 2], static_cast<float>(i) + 200.5f)
                << "count=" << count << " i=" << i;
        }
        // Nothing past 3*count, which is what a mis-sliced mask would hit.
        for (size_t i = 3 * count; i < dst_pred.size(); ++i) {
            EXPECT_FLOAT_EQ(dst_pred[i], sentinel) << "count=" << count << " wrote index " << i;
        }
        // And the two realizations agree everywhere.
        EXPECT_EQ(dst_pred, dst_scal) << "count=" << count;

        // Proof the query is actually consulted rather than one path being
        // taken regardless: the two targets must not produce the same code.
        //
        // Deliberately not an inequality. For a single store the predicated
        // form is slightly *larger* (310 vs 293 bytes on this host): the
        // 3*count lane-bit computation plus three kmovs costs more bytes
        // than an alloca and a compact copy loop. What it buys is not size
        // — it is no stack slot, no per-element copy loop at run time, and
        // about six live GPR values instead of about twenty, which is what
        // made the scalarized path a pool-exhaustion risk.
        EXPECT_NE(size_pred, size_scal) << "count=" << count;
    }
}

// ── Differential: accumulation across a runtime loop ───────────────────
//
// The IR tests above check the shape — one value, two defs, live across
// the loop. This checks the arithmetic, which is what actually goes wrong
// if the accumulator is recorded as a fresh value: the kernel then reads
// the initial zero every trip and stores only the last iteration's
// contribution. That compiles, allocates and runs.

namespace {

struct AccumulateParams {
    const float* a;
    const float* b;
    float* dst;
    size_t count;
};

template <size_t N, bool UseFma>
struct jit_ir_accumulate_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_accumulate_kernel)

    jit_ir_accumulate_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const AccumulateParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const AccumulateParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        set_vec_width(N * sizeof(float) * 8);
        begin_ir();

        auto a = arg<const float*>(&AccumulateParams::a);
        auto b = arg<const float*>(&AccumulateParams::b);
        auto dst = arg<float*>(&AccumulateParams::dst);
        auto count = arg(&AccumulateParams::count);

        // Defined before the loop: the initialization has to be outside,
        // or every iteration starts from zero again.
        auto acc = ir_zero<N>();

        foreach(size_t{0}, count, [&](const variable<size_t>&) {
            auto va = ir_load<N>(a, size_t{0});
            if constexpr (UseFma) {
                auto vb = ir_load<N>(b, size_t{0});
                ir_accumulate(acc, Insn3::fmadd231ps, va, vb);
                ir_advance(b, N * sizeof(float));
            } else {
                ir_accumulate(acc, Insn2::vaddps, va);
            }
            ir_advance(a, N * sizeof(float));
        });

        ir_store<N>(dst, size_t{0}, acc);

        end_ir();
        postamble();
    }
};

template <size_t N, bool UseFma>
void run_accumulate_differential() {
    jit_ir_accumulate_kernel<N, UseFma> kernel;
    kernel.init();

    std::mt19937 rng(20260922);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);

    for (size_t count : {size_t{0}, size_t{1}, size_t{2}, size_t{7}, size_t{33}}) {
        std::vector<float> a(std::max<size_t>(count, 1) * N);
        std::vector<float> b(a.size());
        for (size_t i = 0; i < a.size(); ++i) {
            a[i] = dist(rng);
            b[i] = dist(rng);
        }
        std::vector<float> dst(N, -1.0F);

        AccumulateParams args{a.data(), b.data(), dst.data(), count};
        kernel(args);

        for (size_t lane = 0; lane < N; ++lane) {
            float expected = 0.0F;
            for (size_t i = 0; i < count; ++i) {
                if constexpr (UseFma) {
                    expected = std::fma(a[i * N + lane], b[i * N + lane], expected);
                } else {
                    expected += a[i * N + lane];
                }
            }
            EXPECT_FLOAT_EQ(dst[lane], expected)
                << (UseFma ? "fma" : "add") << " count=" << count << " lane=" << lane;
        }
    }
}

}  // namespace

TEST(JitKernelIR, DifferentialAccumulateAcrossLoop) {
    using namespace dnnl::impl::cpu::x64;
    if (mayiuse(cpu_isa_t::avx512_core)) {
        run_accumulate_differential<16, false>();
        run_accumulate_differential<16, true>();
    } else if (mayiuse(cpu_isa_t::avx2)) {
        run_accumulate_differential<8, false>();
        run_accumulate_differential<8, true>();
    } else {
        GTEST_SKIP() << "requires AVX2 or AVX-512";
    }
}

// ── Type-converting store must not clobber its source ──────────────────
//
// The u8 store narrows f32 -> i32 -> u8. That conversion used to run in
// place on the stored register while the IR only declared it as a read, so
// any later use of the same value saw integers instead of floats. The
// conversion is now its own IR value.

namespace {

struct StoreReuseParams {
    const float* src;
    uint8_t* dst_u8;
    float* dst_f32;
};

template <size_t N>
struct jit_ir_store_reuse_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_store_reuse_kernel)

    jit_ir_store_reuse_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const StoreReuseParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const StoreReuseParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        begin_ir();

        auto src = arg<const float*>(&StoreReuseParams::src);
        auto dst_u8 = arg<uint8_t*>(&StoreReuseParams::dst_u8);
        auto dst_f32 = arg<float*>(&StoreReuseParams::dst_f32);

        auto v = ir_load<N>(src);
        ir_store<N>(dst_u8, v);    // narrowing store, must not touch %v
        ir_store<N>(dst_f32, v);   // same value, still floats

        end_ir();
        postamble();
    }
};

}  // namespace

TEST(JitKernelIR, TypeConvertingStoreKeepsSourceIntact) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "u8 store path requires AVX-512 (vpmovusdb)";
    }
    constexpr size_t N = 16;

    alignas(64) std::array<float, N> src{};
    alignas(64) std::array<uint8_t, N> dst_u8{};
    alignas(64) std::array<float, N> dst_f32{};
    for (size_t i = 0; i < N; ++i) {
        src[i] = static_cast<float>(i) + 0.25f;
    }

    jit_ir_store_reuse_kernel<N> kernel;
    kernel.init();
    kernel(StoreReuseParams{src.data(), dst_u8.data(), dst_f32.data()});

    for (size_t i = 0; i < N; ++i) {
        // f32 store sees the original value, not the converted integers.
        EXPECT_FLOAT_EQ(dst_f32[i], src[i]) << "index " << i;
        EXPECT_EQ(dst_u8[i], static_cast<uint8_t>(std::lround(src[i]))) << "index " << i;
    }
}

// ── Embedded broadcast ─────────────────────────────────────────────────
//
// A splat whose value feeds one instruction should become that
// instruction's memory operand — vfmadd231ps zmm, zmm, m32{1to16} — and a
// splat shared by several must stay in a register. The shape below is a
// GEMM microkernel's: Rows accumulator rows by Cols column blocks, so the
// A splat has exactly Cols uses and the B load has Rows.

namespace {

struct BroadcastFmaParams {
    const float* a;
    const float* b;
    float* dst;
    size_t count;
};

template <size_t N, size_t Rows, size_t Cols>
struct jit_ir_broadcast_fma_kernel : public jit_kernel {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_ir_broadcast_fma_kernel)

    jit_ir_broadcast_fma_kernel() : jit_kernel(jit_name()) {}

    using fn_t = void (*)(const BroadcastFmaParams*);
    fn_t fn_ = nullptr;

    void init() {
        if (create_kernel() != dnnl::impl::status::success) {
            OPENVINO_THROW("Can't generate jit kernel");
        }
        fn_ = (fn_t)(jit_ker());  // NOLINT
    }

    void operator()(const BroadcastFmaParams& args) const { fn_(&args); }

    void generate() override {
        preamble();
        set_vec_width(N * sizeof(float) * 8);
        begin_ir();

        auto a = arg<const float*>(&BroadcastFmaParams::a);
        auto b = arg<const float*>(&BroadcastFmaParams::b);
        auto dst = arg<float*>(&BroadcastFmaParams::dst);
        auto count = arg(&BroadcastFmaParams::count);
        const size_t lda = 64;  // A row stride, in elements

        std::vector<variable<float[N]>> acc;
        acc.reserve(Rows * Cols);
        for (size_t i = 0; i < Rows * Cols; ++i) {
            acc.push_back(ir_zero<N>());
        }

        foreach(size_t{0}, count, [&](const variable<size_t>&) {
            // B first, as the BRGEMM generator records it: the column
            // block is shared by every row, so it is the operand that
            // cannot fold.
            std::vector<variable<float[N]>> col;
            col.reserve(Cols);
            for (size_t c = 0; c < Cols; ++c) {
                col.push_back(ir_load<N>(b, c * N * sizeof(float)));
            }
            for (size_t r = 0; r < Rows; ++r) {
                auto splat = ir_broadcast<N>(a, r * lda * sizeof(float));
                for (size_t c = 0; c < Cols; ++c) {
                    ir_accumulate(acc[r * Cols + c], Insn3::fmadd231ps, splat, col[c]);
                }
            }
            ir_advance(a, sizeof(float));
            ir_advance(b, Cols * N * sizeof(float));
        });

        for (size_t i = 0; i < Rows * Cols; ++i) {
            ir_store<N>(dst, i * N * sizeof(float), acc[i]);
        }

        end_ir();
        postamble();
    }
};

// Same kernel with an injected target, to compare the two forms.
template <size_t N, size_t Rows, size_t Cols>
struct jit_ir_broadcast_fma_target_kernel : public jit_ir_broadcast_fma_kernel<N, Rows, Cols> {
    explicit jit_ir_broadcast_fma_target_kernel(const vector_target& t) { this->set_target(t); }
};

template <size_t N, size_t Rows, size_t Cols, typename Kernel>
void check_broadcast_fma(Kernel& kernel) {
    constexpr size_t lda = 64;
    std::mt19937 rng(20260929 + Cols);
    std::uniform_real_distribution<float> dist(-2.0F, 2.0F);

    for (size_t count : {size_t{0}, size_t{1}, size_t{5}, size_t{17}}) {
        const size_t steps = std::max<size_t>(count, 1);
        std::vector<float> a(Rows * lda + steps);
        std::vector<float> b(steps * Cols * N);
        for (auto& v : a) {
            v = dist(rng);
        }
        for (auto& v : b) {
            v = dist(rng);
        }
        std::vector<float> dst(Rows * Cols * N, -1.0F);

        kernel(BroadcastFmaParams{a.data(), b.data(), dst.data(), count});

        for (size_t r = 0; r < Rows; ++r) {
            for (size_t c = 0; c < Cols; ++c) {
                for (size_t lane = 0; lane < N; ++lane) {
                    float expected = 0.0F;
                    for (size_t k = 0; k < count; ++k) {
                        expected = std::fma(a[r * lda + k],
                                            b[k * Cols * N + c * N + lane],
                                            expected);
                    }
                    EXPECT_FLOAT_EQ(dst[(r * Cols + c) * N + lane], expected)
                        << "count " << count << " row " << r << " col " << c << " lane " << lane;
                }
            }
        }
    }
}

}  // namespace

TEST(JitKernelIR, DifferentialBroadcastFoldedIntoFma) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "embedded broadcast is an AVX-512 encoding";
    }
    constexpr size_t N = 16;

    // Cols == 1: the splat has one use and folds.
    jit_ir_broadcast_fma_kernel<N, 2, 1> folded;
    folded.init();
    check_broadcast_fma<N, 2, 1>(folded);

    // Cols == 3: three uses, so it must stay in a register.
    jit_ir_broadcast_fma_kernel<N, 2, 3> shared;
    shared.init();
    check_broadcast_fma<N, 2, 3>(shared);
}

TEST(JitKernelIR, FoldingTheBroadcastRemovesInstructions) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "embedded broadcast is an AVX-512 encoding";
    }
    // The only test here that asserts a fold *happened* through the real
    // pipeline rather than by calling the pass, so it is the only one the
    // A/B switch can turn off.
    if (std::getenv("OV_JIT_IR_NO_FOLD") != nullptr) {
        GTEST_SKIP() << "folding is disabled by OV_JIT_IR_NO_FOLD";
    }
    constexpr size_t N = 16;

    // One use per splat, so the target with the capability folds and the
    // AVX2-shaped one cannot. Both targets are injected, so the only
    // difference between the two kernels is the fold — comparing against
    // the host target instead would also pick up OV_JIT_IR_LOOP_ALIGN.
    const host_like_target with_bcast;
    const no_broadcast_operand_target no_bcast;

    jit_ir_broadcast_fma_target_kernel<N, 2, 1> folded(with_bcast);
    folded.init();
    check_broadcast_fma<N, 2, 1>(folded);

    jit_ir_broadcast_fma_target_kernel<N, 2, 1> unfolded(no_bcast);
    unfolded.init();
    check_broadcast_fma<N, 2, 1>(unfolded);

    EXPECT_LT(folded.getSize(), unfolded.getSize())
        << "folding a single-use splat into the FMA should shorten the kernel";
}

TEST(JitKernelIR, SharedBroadcastIsUnaffectedByTheTarget) {
    using namespace dnnl::impl::cpu::x64;
    if (!mayiuse(cpu_isa_t::avx512_core)) {
        GTEST_SKIP() << "embedded broadcast is an AVX-512 encoding";
    }
    constexpr size_t N = 16;

    // Three uses: nothing to fold, so the capability makes no difference.
    // This is the case the BRGEMM generator hits on a wide N, and it is
    // what keeps one load from becoming three.
    const host_like_target with_bcast;
    const no_broadcast_operand_target no_bcast;

    jit_ir_broadcast_fma_target_kernel<N, 2, 3> with(with_bcast);
    with.init();

    jit_ir_broadcast_fma_target_kernel<N, 2, 3> without(no_bcast);
    without.init();

    EXPECT_EQ(with.getSize(), without.getSize());
}

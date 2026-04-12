// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_ir.hpp"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <ostream>
#include <string>
#include <vector>

namespace ov::intel_cpu::jit_kernel_ir {

namespace {

// Recursive tree walk for interval computation. Processes ops in order,
// assigning sequential indices. When a loop body is encountered, its ops
// are walked inline and all intervals overlapping the loop range are
// extended to cover the full body.
void compute_intervals_impl(const std::vector<Op>& ops,
                            std::vector<Interval>& intervals,
                            std::uint32_t& index) {
    for (const auto& op : ops) {
        if (op.body) {
            // Region op: walk body. For loops, extend intervals.
            auto region_begin = index;
            compute_intervals_impl(op.body->ops(), intervals, index);
            auto region_end = (index > 0) ? index - 1 : 0u;

            if (op.is_loop) {
                // Loop: any interval overlapping [begin, end] must cover
                // the full body (the loop may iterate).
                for (auto& iv : intervals) {
                    if (iv.start == std::numeric_limits<std::uint32_t>::max()) continue;
                    if (iv.start <= region_end && iv.end >= region_begin) {
                        if (iv.end < region_end) {
                            iv.end = region_end;
                        }
                    }
                }
            }
            // Branches: no extension — each branch executes once,
            // sequential indexing handles interval computation naturally.
        } else {
            // Regular op
            for (value_id read : op.reads) {
                if (intervals[read].end < index) {
                    intervals[read].end = index;
                }
            }
            if (op.def != invalid_value) {
                intervals[op.def].start = index;
                if (intervals[op.def].end < index) {
                    intervals[op.def].end = index;
                }
            }
            ++index;
        }
    }
}

}  // namespace

std::vector<Interval> compute_intervals(const IR& ir) {
    std::vector<Interval> intervals(ir.value_count());

    for (value_id v = 0; v < ir.value_count(); ++v) {
        intervals[v].id = v;
        intervals[v].start = std::numeric_limits<std::uint32_t>::max();
        intervals[v].end = 0;
    }

    std::uint32_t index = 0;
    compute_intervals_impl(ir.ops(), intervals, index);
    return intervals;
}

Assignment linear_scan(const std::vector<Interval>& intervals,
                       std::uint32_t pool_size) {
    // Sort intervals by start op index. Intervals that never got a def
    // (sentinel start == max) sort to the end and are skipped.
    std::vector<const Interval*> order;
    order.reserve(intervals.size());
    for (const auto& iv : intervals) {
        if (iv.start != std::numeric_limits<std::uint32_t>::max()) {
            order.push_back(&iv);
        }
    }
    std::sort(order.begin(), order.end(), [](const Interval* a, const Interval* b) {
        if (a->start != b->start) {
            return a->start < b->start;
        }
        return a->id < b->id;
    });

    // Free list of physical register indices. Pop front = "youngest freed".
    // Later slices may swap this for a recency-biased policy.
    std::deque<std::uint32_t> free_list;
    for (std::uint32_t r = 0; r < pool_size; ++r) {
        free_list.push_back(r);
    }

    // Active set: intervals currently holding a register. Kept sorted by end
    // ascending so expire() is a linear scan of the front. For Slice 1's
    // scale (dozens of intervals) linear is fine; promote to a heap later if
    // the profile says so.
    struct ActiveEntry {
        const Interval* iv = nullptr;
        PhysReg reg{};
    };
    std::vector<ActiveEntry> active;

    Assignment result;
    result.reg.reserve(order.size());

    auto expire = [&](std::uint32_t current_start) {
        // Free any interval whose end is strictly before the new interval's
        // start. Equal-end is NOT expired — a value used at op i is still
        // live at op i and conflicts with another def at op i.
        auto it = active.begin();
        while (it != active.end()) {
            if (it->iv->end < current_start) {
                free_list.push_back(it->reg.idx);
                it = active.erase(it);
            } else {
                ++it;
            }
        }
    };

    auto insert_active = [&](const Interval* iv, PhysReg reg) {
        // Keep active sorted by end ascending for cheap front-expiry.
        auto pos = std::upper_bound(active.begin(), active.end(), iv->end,
                                    [](std::uint32_t end, const ActiveEntry& e) {
                                        return end < e.iv->end;
                                    });
        active.insert(pos, ActiveEntry{iv, reg});
    };

    for (const Interval* iv : order) {
        expire(iv->start);

        if (free_list.empty()) {
            throw allocation_failure(
                "jit_kernel_ir::linear_scan: pool of " + std::to_string(pool_size) +
                " registers exhausted at op index " + std::to_string(iv->start) +
                " (value id " + std::to_string(iv->id) +
                "); Slice 1 has no spill support");
        }

        PhysReg reg{free_list.front()};
        free_list.pop_front();
        result.reg.emplace(iv->id, reg);
        insert_active(iv, reg);

        if (active.size() > result.peak_live) {
            result.peak_live = static_cast<std::uint32_t>(active.size());
        }
    }

    return result;
}

void IR::dump(std::ostream& os) const {
    dump_ops(os, *this);
}

void dump_ops(std::ostream& os, const IR& ir) {
    const auto& ops = ir.ops();
    for (std::uint32_t i = 0; i < ops.size(); ++i) {
        const Op& op = ops[i];
        os << "  " << i << ": ";
        if (op.def != invalid_value) {
            os << "%" << op.def << " = ";
        } else {
            os << "       ";
        }
        os << (op.is_copy ? "copy" : "op") << "(";
        for (std::size_t r = 0; r < op.reads.size(); ++r) {
            if (r != 0) {
                os << ", ";
            }
            os << "%" << op.reads[r];
        }
        os << ")\n";
    }
}

void dump_assignment(std::ostream& os,
                     const std::vector<Interval>& intervals,
                     const Assignment& assignment) {
    for (const auto& iv : intervals) {
        if (iv.start == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        os << "  %" << iv.id << ": [" << iv.start << ", " << iv.end << "] -> ";
        auto it = assignment.reg.find(iv.id);
        if (it == assignment.reg.end()) {
            os << "unassigned";
        } else {
            os << "p" << it->second.idx;
        }
        os << "\n";
    }
    os << "  peak_live = " << assignment.peak_live << "\n";
}

}  // namespace ov::intel_cpu::jit_kernel_ir

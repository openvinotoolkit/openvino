// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_ir.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

namespace ov::intel_cpu::jit_kernel_ir {

namespace {

// Recursive tree walk for interval computation. Processes ops in order,
// assigning sequential indices. When a loop body is encountered, its ops
// are walked inline and all intervals overlapping the loop range are
// extended to cover the full body.
void compute_intervals_impl(const std::list<Op>& ops,
                            std::vector<Interval>& intervals,
                            std::uint32_t& index) {
    for (const auto& op : ops) {
        if (op.body) {
            // Region op: walk body. For loops, extend intervals.
            auto region_begin = index;
            compute_intervals_impl(op.body->ops(), intervals, index);
            auto region_end = (index > 0) ? index - 1 : 0u;

            if (op.is_loop) {
                // Loop: values defined BEFORE the loop but used INSIDE
                // must stay live through the entire body (the loop iterates
                // and each iteration re-reads the value). Values defined
                // inside the loop are SSA-fresh each iteration — they keep
                // their natural intra-iteration lifetime.
                for (auto& iv : intervals) {
                    if (iv.start == std::numeric_limits<std::uint32_t>::max()) continue;
                    if (iv.start < region_begin && iv.end >= region_begin &&
                        iv.end < region_end) {
                        iv.end = region_end;
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
                if (op.is_copy && op.reads.size() == 1) {
                    intervals[op.def].copy_of = op.reads[0];
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

namespace {

// Build a map: value_id → Op* (the defining op) by walking the IR tree.
// Also collects which ops are rematerializable (empty reads, not a region).
void collect_def_ops(std::list<Op>& ops,
                     std::unordered_map<value_id, Op*>& def_map) {
    for (auto& op : ops) {
        if (op.def != invalid_value) {
            def_map[op.def] = &op;
        }
        if (op.body) {
            collect_def_ops(op.body->ops(), def_map);
        }
    }
}

// Find the list position and iterator of the op that first reads `vid`
// at or after op index `after`. Walks the tree linearly.
struct UseLocation {
    std::list<Op>* parent_list = nullptr;
    std::list<Op>::iterator it;
    std::uint32_t index = 0;
};

bool find_next_use(std::list<Op>& ops, value_id vid, std::uint32_t after,
                   std::uint32_t& index, UseLocation& result) {
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        auto& op = *it;
        if (op.body) {
            if (find_next_use(op.body->ops(), vid, after, index, result))
                return true;
        } else {
            if (index >= after) {
                for (auto read : op.reads) {
                    if (read == vid) {
                        result = {&ops, it, index};
                        return true;
                    }
                }
            }
            ++index;
        }
    }
    return false;
}

// Recursively rewrite all reads of `old_id` → `new_id` in an op list,
// including nested REGION bodies. Returns the highest op index where
// a rewrite occurred, or `last_read` unchanged if no rewrites happen.
void rewrite_reads_recursive(std::list<Op>& ops, value_id old_id,
                             value_id new_id, std::uint32_t& scan_idx,
                             std::uint32_t& last_read) {
    for (auto& op : ops) {
        if (op.body) {
            rewrite_reads_recursive(op.body->ops(), old_id, new_id,
                                    scan_idx, last_read);
        } else {
            for (auto& read : op.reads) {
                if (read == old_id) {
                    read = new_id;
                    last_read = scan_idx;
                }
            }
            ++scan_idx;
        }
    }
}

}  // namespace

Assignment linear_scan(IR& ir,
                       std::vector<Interval>& intervals,
                       std::uint32_t pool_size) {
    // Build def map for rematerialization lookup
    std::unordered_map<value_id, Op*> def_map;
    collect_def_ops(ir.ops(), def_map);

    // Use indices into intervals[] — safe across push_back/reallocation.
    std::vector<std::size_t> order;
    order.reserve(intervals.size());
    for (std::size_t idx = 0; idx < intervals.size(); ++idx) {
        if (intervals[idx].start != std::numeric_limits<std::uint32_t>::max()) {
            order.push_back(idx);
        }
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (intervals[a].start != intervals[b].start) {
            return intervals[a].start < intervals[b].start;
        }
        return intervals[a].id < intervals[b].id;
    });

    std::deque<std::uint32_t> free_list;
    for (std::uint32_t r = 0; r < pool_size; ++r) {
        free_list.push_back(r);
    }

    struct ActiveEntry {
        std::size_t iv_idx = 0;   // index into intervals[]
        PhysReg reg{};
    };
    std::vector<ActiveEntry> active;

    Assignment result;
    result.reg.reserve(order.size());

    auto expire = [&](std::uint32_t current_start) {
        auto it = active.begin();
        while (it != active.end()) {
            if (intervals[it->iv_idx].end < current_start) {
                free_list.push_back(it->reg.idx);
                it = active.erase(it);
            } else {
                ++it;
            }
        }
    };

    auto insert_active = [&](std::size_t iv_idx, PhysReg reg) {
        auto end = intervals[iv_idx].end;
        auto pos = std::upper_bound(active.begin(), active.end(), end,
                                    [&](std::uint32_t e, const ActiveEntry& ae) {
                                        return e < intervals[ae.iv_idx].end;
                                    });
        active.insert(pos, ActiveEntry{iv_idx, reg});
    };

    for (std::size_t i = 0; i < order.size(); ++i) {
        auto iv_idx = order[i];
        // Note: do NOT hold a reference to intervals[iv_idx] across
        // the remat block — push_back can reallocate the vector.
        expire(intervals[iv_idx].start);

        // Trivial coalescing
        if (intervals[iv_idx].copy_of != invalid_value) {
            auto src_it = result.reg.find(intervals[iv_idx].copy_of);
            if (src_it != result.reg.end()) {
                auto fl_it = std::find(free_list.begin(), free_list.end(), src_it->second.idx);
                if (fl_it != free_list.end()) {
                    free_list.erase(fl_it);
                    result.reg.emplace(intervals[iv_idx].id, src_it->second);
                    insert_active(iv_idx, src_it->second);
                    continue;
                }
            }
        }

        // Rematerialization: if pool exhausted, evict a remat-able victim
        if (free_list.empty()) {
            // Collect rematerializable candidates sorted by furthest end
            std::vector<std::size_t> candidates;
            for (std::size_t a = 0; a < active.size(); ++a) {
                auto vid = intervals[active[a].iv_idx].id;
                auto def_it = def_map.find(vid);
                if (def_it == def_map.end()) continue;
                if (!def_it->second->reads.empty()) continue;
                candidates.push_back(a);
            }
            std::sort(candidates.begin(), candidates.end(),
                      [&](std::size_t a, std::size_t b) {
                          return intervals[active[a].iv_idx].end > intervals[active[b].iv_idx].end;
                      });

            bool evicted = false;
            for (auto victim_pos : candidates) {
                auto& victim = active[victim_pos];
                auto victim_id = intervals[victim.iv_idx].id;
                auto* victim_def = def_map[victim_id];

                // Find victim's next use after eviction point.
                // If no forward use, retry from the beginning — the value
                // may be used earlier in a loop body that wraps around.
                UseLocation use_loc{};
                std::uint32_t search_idx = 0;
                find_next_use(ir.ops(), victim_id, intervals[iv_idx].start, search_idx, use_loc);

                if (!use_loc.parent_list) {
                    search_idx = 0;
                    find_next_use(ir.ops(), victim_id, 0, search_idx, use_loc);
                }

                if (!use_loc.parent_list) {
                    // No use found at all — value is dead, skip.
                    continue;
                }

                auto victim_reg = victim.reg;

                // Truncate victim's interval at the eviction point
                intervals[victim.iv_idx].end = intervals[iv_idx].start;

                // Clone: insert a remat op before the use site
                auto new_id = static_cast<value_id>(intervals.size());
                ir.set_value_count(new_id + 1);

                Op remat_op;
                remat_op.def = new_id;
                remat_op.emit = victim_def->emit;
                remat_op.name = "remat";

                use_loc.parent_list->insert(use_loc.it, std::move(remat_op));

                // Rewrite reads: victim_id → new_id from use_loc onward,
                // including inside nested REGION bodies (if/else branches).
                // Track the last read index to compute the clone's actual end.
                // The remat op was inserted before use_loc.it, so use_loc.it
                // is now at index use_loc.index + 1.
                std::uint32_t clone_last_read = use_loc.index;
                {
                    std::uint32_t scan_idx = use_loc.index + 1;
                    for (auto it = use_loc.it; it != use_loc.parent_list->end(); ++it) {
                        if (it->body) {
                            rewrite_reads_recursive(it->body->ops(), victim_id,
                                                    new_id, scan_idx,
                                                    clone_last_read);
                        } else {
                            for (auto& read : it->reads) {
                                if (read == victim_id) {
                                    read = new_id;
                                    clone_last_read = scan_idx;
                                }
                            }
                            ++scan_idx;
                        }
                    }
                }

                // Create interval for the clone — use actual last read,
                // not the original's end. This keeps the clone short-lived
                // (especially inside loops where the original was extended).
                Interval remat_iv;
                remat_iv.id = new_id;
                remat_iv.start = use_loc.index;
                remat_iv.end = clone_last_read;
                intervals.push_back(remat_iv);
                auto new_iv_idx = intervals.size() - 1;

                // Insert into sorted order for later processing
                auto insert_pos = std::upper_bound(
                    order.begin() + static_cast<long>(i) + 1, order.end(),
                    new_iv_idx,
                    [&](std::size_t a, std::size_t b) {
                        return intervals[a].start < intervals[b].start ||
                               (intervals[a].start == intervals[b].start &&
                                intervals[a].id < intervals[b].id);
                    });
                order.insert(insert_pos, new_iv_idx);

                // Remove victim from active, free its register
                active.erase(active.begin() + static_cast<long>(victim_pos));
                free_list.push_back(victim_reg.idx);
                evicted = true;
                break;
            }

            if (!evicted) {
                throw allocation_failure(
                    "jit_kernel_ir::linear_scan: pool of " + std::to_string(pool_size) +
                    " registers exhausted at op index " + std::to_string(intervals[iv_idx].start) +
                    " (value id " + std::to_string(intervals[iv_idx].id) +
                    "); no rematerializable victim found");
            }
        }

        PhysReg reg{free_list.front()};
        free_list.pop_front();
        // Re-fetch from intervals[] — remat may have push_back'd,
        // invalidating the earlier `iv` reference.
        result.reg.emplace(intervals[iv_idx].id, reg);
        insert_active(iv_idx, reg);

        if (active.size() > result.peak_live) {
            result.peak_live = static_cast<std::uint32_t>(active.size());
        }
    }

    return result;
}

void IR::dump(std::ostream& os) const {
    dump_ops(os, *this);
}

namespace {
void dump_ops_impl(std::ostream& os, const std::list<Op>& ops, std::uint32_t& i, int depth) {
    std::string indent(static_cast<std::size_t>(depth) * 2 + 2, ' ');
    for (const auto& op : ops) {
        if (op.body) {
            os << indent << (op.is_loop ? "LOOP {\n" : "REGION {\n");
            dump_ops_impl(os, op.body->ops(), i, depth + 1);
            os << indent << "}\n";
        } else {
            os << indent << i << ": ";
            if (op.def != invalid_value) {
                os << "%" << op.def << " = ";
            } else {
                os << "       ";
            }
            const char* tag = op.is_copy ? "copy" : (op.name[0] ? op.name : "op");
            os << tag << "(";
            for (std::size_t r = 0; r < op.reads.size(); ++r) {
                if (r != 0) os << ", ";
                os << "%" << op.reads[r];
            }
            os << ")\n";
            ++i;
        }
    }
}
}  // namespace

void dump_ops(std::ostream& os, const IR& ir) {
    std::uint32_t i = 0;
    dump_ops_impl(os, ir.ops(), i, 0);
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

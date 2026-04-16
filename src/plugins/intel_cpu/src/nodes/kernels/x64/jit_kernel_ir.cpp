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
#include <sstream>
#include <string>
#include <vector>

namespace ov::intel_cpu::jit_kernel_ir {

namespace {

bool env_enabled(const char* name) {
    return std::getenv(name) != nullptr;
}

struct remat_debug_config {
    bool disable_remat = env_enabled("OV_JIT_IR_DISABLE_REMAT");
    bool disable_remat_in_regions = env_enabled("OV_JIT_IR_DISABLE_REMAT_IN_REGIONS");
    bool disable_remat_wraparound = env_enabled("OV_JIT_IR_DISABLE_REMAT_WRAPAROUND");
    bool remat_single_use = env_enabled("OV_JIT_IR_REMAT_SINGLE_USE");
    bool trace_remat = env_enabled("OV_JIT_IR_TRACE_REMAT");
    bool trace_all = env_enabled("OV_JIT_IR_TRACE");
};

const remat_debug_config& get_remat_debug_config() {
    static const remat_debug_config cfg{};
    return cfg;
}

void trace_ir(const std::string& msg) {
    if (!get_remat_debug_config().trace_all) {
        return;
    }
    std::cout << "[jit_kernel_ir] " << msg << "\n";
}

std::string format_reads(const std::vector<value_id>& reads) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < reads.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << "%" << reads[i];
    }
    os << "]";
    return os.str();
}

void trace_remat_event(const char* phase,
                       value_id victim_id,
                       value_id clone_id,
                       std::uint32_t evict_at,
                       std::uint32_t insert_at,
                       std::uint32_t rewrite_begin,
                       std::uint32_t rewrite_end,
                       bool wrapped,
                       bool in_region) {
    if (!get_remat_debug_config().trace_remat) {
        return;
    }
    std::cout << "[jit_kernel_ir] remat " << phase
              << " victim=%" << victim_id
              << " clone=%" << clone_id
              << " evict_at=" << evict_at
              << " insert_at=" << insert_at
              << " rewrite_begin=" << rewrite_begin
              << " rewrite_end=";
    if (rewrite_end == std::numeric_limits<std::uint32_t>::max()) {
        std::cout << "end";
    } else {
        std::cout << rewrite_end;
    }
    std::cout << " wrapped=" << wrapped
              << " in_region=" << in_region
              << "\n";
}

// Per-recursion-level local segment tracking for a single value.
struct LocalSeg {
    std::uint32_t first;
    std::uint32_t last;
};

// Recursive tree walk for live range computation. Each recursion level
// tracks its own local segments. On return, local segments are flushed
// into the LiveRange via addSegment(). Sibling branch bodies (separate
// recursive calls) naturally produce separate segments.
void compute_live_ranges_impl(const std::list<Op>& ops,
                              std::vector<LiveRange>& ranges,
                              std::uint32_t& index) {
    std::unordered_map<value_id, LocalSeg> local;

    for (const auto& op : ops) {
        if (op.body) {
            auto region_begin = index;
            trace_ir(std::string("ranges enter ") + (op.is_loop ? "loop" : "region") +
                     " begin=" + std::to_string(region_begin));
            compute_live_ranges_impl(op.body->ops(), ranges, index);
            auto region_end = (index > 0) ? index - 1 : 0u;
            trace_ir(std::string("ranges exit ") + (op.is_loop ? "loop" : "region") +
                     " begin=" + std::to_string(region_begin) +
                     " end=" + std::to_string(region_end));

            if (op.is_loop) {
                // Loop: values defined BEFORE the loop but used INSIDE
                // must stay live through the entire body.

                // Extend committed segments from child recursions.
                for (auto& lr : ranges) {
                    if (lr.empty()) continue;
                    if (lr.beginIndex() >= region_begin) continue;
                    for (auto& seg : lr.segments) {
                        if (seg.end >= region_begin && seg.end < region_end) {
                            trace_ir("extend %" + std::to_string(lr.id) +
                                     " loop_end " + std::to_string(seg.end) +
                                     " -> " + std::to_string(region_end));
                            seg.end = region_end;
                        }
                    }
                }

                // Extend local tracking for values defined at this level
                // and used inside the loop body.
                for (auto& [vid, seg] : local) {
                    if (seg.first < region_begin) {
                        for (const auto& committed : ranges[vid].segments) {
                            if (committed.start >= region_begin &&
                                committed.start <= region_end) {
                                seg.last = std::max(seg.last, region_end);
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            trace_ir("visit op@" + std::to_string(index) +
                     " def=" + (op.def == invalid_value ? std::string("-") : "%" + std::to_string(op.def)) +
                     " reads=" + format_reads(op.reads) +
                     (op.name[0] ? " name=" + std::string(op.name) : ""));
            for (value_id read : op.reads) {
                auto it = local.find(read);
                if (it != local.end()) {
                    if (index > it->second.last) {
                        trace_ir("  last_use %" + std::to_string(read) +
                                 " " + std::to_string(it->second.last) +
                                 " -> " + std::to_string(index));
                        it->second.last = index;
                    }
                } else {
                    trace_ir("  first_use %" + std::to_string(read) +
                             " at " + std::to_string(index));
                    local[read] = {index, index};
                }
            }
            if (op.def != invalid_value) {
                local[op.def] = {index, index};
                if (op.is_copy && op.reads.size() == 1) {
                    ranges[op.def].copy_of = op.reads[0];
                }
            }
            ++index;
        }
    }

    // Flush local segments into LiveRanges.
    for (const auto& [vid, seg] : local) {
        ranges[vid].addSegment({seg.first, seg.last});
    }
}

}  // namespace

// ── LiveRange methods ─────────────────────────────────────────────────

std::uint32_t LiveRange::beginIndex() const noexcept {
    return segments.empty() ? std::numeric_limits<std::uint32_t>::max()
                            : segments.front().start;
}

std::uint32_t LiveRange::endIndex() const noexcept {
    return segments.empty() ? 0u : segments.back().end;
}

bool LiveRange::empty() const noexcept {
    return segments.empty();
}

bool LiveRange::liveAt(std::uint32_t index) const noexcept {
    if (segments.empty()) return false;
    // Binary search: find last segment with start <= index.
    auto it = std::upper_bound(segments.begin(), segments.end(), index,
                               [](std::uint32_t val, const Segment& seg) {
                                   return val < seg.start;
                               });
    if (it == segments.begin()) return false;
    --it;
    return index <= it->end;
}

void LiveRange::addSegment(Segment s) {
    if (segments.empty()) {
        segments.push_back(s);
        return;
    }

    // Find insertion point: first segment whose start > s.start.
    auto it = std::upper_bound(segments.begin(), segments.end(), s.start,
                               [](std::uint32_t val, const Segment& seg) {
                                   return val < seg.start;
                               });
    it = segments.insert(it, s);

    // Merge with predecessor if overlapping (NOT merely adjacent).
    if (it != segments.begin()) {
        auto prev = std::prev(it);
        if (prev->end >= it->start) {
            prev->end = std::max(prev->end, it->end);
            it = segments.erase(it);
            it = prev;
        }
    }

    // Merge with successors if overlapping.
    while (std::next(it) != segments.end()) {
        auto next_it = std::next(it);
        if (it->end >= next_it->start) {
            it->end = std::max(it->end, next_it->end);
            segments.erase(next_it);
        } else {
            break;
        }
    }
}

// ── compute_live_ranges ───────────────────────────────────────────────

std::vector<LiveRange> compute_live_ranges(const IR& ir) {
    std::vector<LiveRange> ranges(ir.value_count());

    for (value_id v = 0; v < ir.value_count(); ++v) {
        ranges[v].id = v;
    }

    std::uint32_t index = 0;
    compute_live_ranges_impl(ir.ops(), ranges, index);
    return ranges;
}

namespace {

struct PressureEntry {
    std::size_t lr_idx = 0;
};

template <typename Entry>
void insert_by_end(std::vector<Entry>& active,
                   const std::vector<LiveRange>& ranges,
                   std::size_t lr_idx) {
    auto end = ranges[lr_idx].endIndex();
    auto pos = std::upper_bound(active.begin(), active.end(), end,
                                [&](std::uint32_t e, const Entry& ae) {
                                    return e < ranges[ae.lr_idx].endIndex();
                                });
    active.insert(pos, Entry{lr_idx});
}

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

bool find_next_use_impl(std::list<Op>& ops, value_id vid, std::uint32_t after,
                        std::uint32_t& index, UseLocation& result) {
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        auto& op = *it;
        if (op.body) {
            if (find_next_use_impl(op.body->ops(), vid, after, index, result)) {
                return true;
            }
        } else {
            if (index >= after) {
                for (auto read : op.reads) {
                    if (read == vid) {
                        result.parent_list = &ops;
                        result.it = it;
                        result.index = index;
                        return true;
                    }
                }
            }
            ++index;
        }
    }
    return false;
}

bool find_next_use(std::list<Op>& ops, value_id vid, std::uint32_t after,
                   std::uint32_t& index, UseLocation& result) {
    return find_next_use_impl(ops, vid, after, index, result);
}

std::uint32_t count_linear_ops(const std::list<Op>& ops) {
    std::uint32_t count = 0;
    for (const auto& op : ops) {
        if (op.body) {
            count += count_linear_ops(op.body->ops());
        } else {
            ++count;
        }
    }
    return count;
}

std::uint32_t rewrite_reads_same_list_suffix(std::list<Op>& ops,
                                             std::list<Op>::iterator first,
                                             std::uint32_t first_index,
                                             value_id old_id,
                                             value_id new_id) {
    std::uint32_t current_index = first_index;
    std::uint32_t last_read = first_index;
    for (auto it = first; it != ops.end(); ++it) {
        if (it->body) {
            current_index += count_linear_ops(it->body->ops());
            continue;
        }
        bool rewritten = false;
        for (auto& read : it->reads) {
            if (read == old_id) {
                read = new_id;
                rewritten = true;
            }
        }
        if (rewritten) {
            last_read = current_index;
        }
        ++current_index;
    }
    return last_read;
}

}  // namespace

bool rematerialize_for_pressure(IR& ir,
                                const std::vector<LiveRange>& ranges,
                                std::uint32_t pool_size) {
    if (get_remat_debug_config().disable_remat) {
        return false;
    }

    std::unordered_map<value_id, Op*> def_map;
    collect_def_ops(ir.ops(), def_map);

    std::vector<std::size_t> order;
    order.reserve(ranges.size());
    for (std::size_t idx = 0; idx < ranges.size(); ++idx) {
        if (!ranges[idx].empty()) {
            order.push_back(idx);
        }
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (ranges[a].beginIndex() != ranges[b].beginIndex()) {
            return ranges[a].beginIndex() < ranges[b].beginIndex();
        }
        return ranges[a].id < ranges[b].id;
    });

    std::vector<PressureEntry> active;
    auto expire = [&](std::uint32_t current_start) {
        auto it = active.begin();
        while (it != active.end()) {
            if (ranges[it->lr_idx].endIndex() < current_start) {
                it = active.erase(it);
            } else {
                ++it;
            }
        }
    };

    for (auto lr_idx : order) {
        expire(ranges[lr_idx].beginIndex());
        if (active.size() < pool_size) {
            insert_by_end(active, ranges, lr_idx);
            continue;
        }

        std::vector<std::size_t> candidates;
        for (std::size_t pos = 0; pos < active.size(); ++pos) {
            const auto vid = ranges[active[pos].lr_idx].id;
            auto def_it = def_map.find(vid);
            if (def_it == def_map.end()) continue;
            if (!def_it->second->reads.empty()) continue;
            candidates.push_back(pos);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [&](std::size_t a, std::size_t b) {
                      return ranges[active[a].lr_idx].endIndex() > ranges[active[b].lr_idx].endIndex();
                  });

        for (auto victim_pos : candidates) {
            const auto victim_lr_idx = active[victim_pos].lr_idx;
            const auto victim_id = ranges[victim_lr_idx].id;
            auto* victim_def = def_map[victim_id];

            UseLocation use_loc{};
            std::uint32_t search_idx = 0;
            find_next_use(ir.ops(), victim_id, ranges[lr_idx].beginIndex(), search_idx, use_loc);
            if (!use_loc.parent_list) {
                continue;
            }

            const auto new_id = static_cast<value_id>(ir.value_count());
            ir.set_value_count(new_id + 1);

            Op remat_op;
            remat_op.def = new_id;
            remat_op.emit = victim_def->emit;
            remat_op.name = "remat";
            use_loc.parent_list->insert(use_loc.it, std::move(remat_op));

            const auto rewrite_end = get_remat_debug_config().remat_single_use
                ? use_loc.index
                : rewrite_reads_same_list_suffix(*use_loc.parent_list,
                                                 use_loc.it,
                                                 use_loc.index,
                                                 victim_id,
                                                 new_id);
            if (get_remat_debug_config().remat_single_use) {
                for (auto& read : use_loc.it->reads) {
                    if (read == victim_id) {
                        read = new_id;
                    }
                }
            }

            trace_remat_event("repair",
                              victim_id,
                              new_id,
                              ranges[lr_idx].beginIndex(),
                              use_loc.index,
                              use_loc.index,
                              rewrite_end,
                              false,
                              false);
            return true;
        }

        return false;
    }

    return false;
}

Assignment linear_scan(IR& /*ir*/,
                       std::vector<LiveRange>& ranges,
                       std::uint32_t pool_size) {
    // LLVM-style per-register interference allocation. Each physical
    // register maintains a segment union — the merged list of all segments
    // assigned to it. A LiveRange can use a register iff none of its
    // segments overlap any segment already on that register. Two values
    // with non-overlapping segments (e.g., branch-local values) naturally
    // share a register.
    //
    // No expire, no temp_free, no re-acquisition. One register per value,
    // assigned once, correct by construction.

    // Sort LiveRanges by beginIndex (standard linear-scan order).
    std::vector<std::size_t> order;
    order.reserve(ranges.size());
    for (std::size_t idx = 0; idx < ranges.size(); ++idx) {
        if (!ranges[idx].empty()) {
            order.push_back(idx);
        }
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (ranges[a].beginIndex() != ranges[b].beginIndex()) {
            return ranges[a].beginIndex() < ranges[b].beginIndex();
        }
        return ranges[a].id < ranges[b].id;
    });

    // Per-register segment union: all segments assigned to each physical reg.
    std::vector<std::vector<Segment>> reg_segments(pool_size);

    // Check if any segment of `lr` overlaps any segment on register `r`.
    auto interferes = [&](const LiveRange& lr, std::uint32_t r) -> bool {
        for (const auto& s : lr.segments) {
            for (const auto& e : reg_segments[r]) {
                if (s.start <= e.end && e.start <= s.end) {
                    return true;
                }
            }
        }
        return false;
    };

    // Add all segments of `lr` to register `r`'s union.
    auto add_to_reg = [&](const LiveRange& lr, std::uint32_t r) {
        for (const auto& s : lr.segments) {
            reg_segments[r].push_back(s);
        }
    };

    Assignment result;
    result.reg.reserve(order.size());

    for (auto lr_idx : order) {
        const auto& lr = ranges[lr_idx];
        auto vid = lr.id;

        trace_ir("alloc %" + std::to_string(vid) +
                 " segs=" + std::to_string(lr.segments.size()) +
                 " [" + std::to_string(lr.beginIndex()) +
                 "," + std::to_string(lr.endIndex()) + "]");

        // Trivial coalescing: prefer the source's register if compatible.
        if (lr.copy_of != invalid_value) {
            auto src_it = result.reg.find(lr.copy_of);
            if (src_it != result.reg.end() && !interferes(lr, src_it->second.idx)) {
                trace_ir("  coalesce %" + std::to_string(vid) +
                         " with %" + std::to_string(lr.copy_of) +
                         " on p" + std::to_string(src_it->second.idx));
                result.reg.emplace(vid, src_it->second);
                add_to_reg(lr, src_it->second.idx);
                continue;
            }
        }

        // Find first non-interfering register.
        bool assigned = false;
        for (std::uint32_t r = 0; r < pool_size; ++r) {
            if (!interferes(lr, r)) {
                trace_ir("  assign %" + std::to_string(vid) +
                         " -> p" + std::to_string(r));
                result.reg.emplace(vid, PhysReg{r});
                add_to_reg(lr, r);
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            throw allocation_failure(
                "jit_kernel_ir::linear_scan: pool of " + std::to_string(pool_size) +
                " registers exhausted for value %" + std::to_string(vid) +
                " at op index " + std::to_string(lr.beginIndex()));
        }
    }

    // Compute peak_live: max number of registers simultaneously in use.
    // Walk all op indices and count how many values are live at each.
    std::uint32_t max_idx = 0;
    for (const auto& lr : ranges) {
        if (!lr.empty() && lr.endIndex() > max_idx) {
            max_idx = lr.endIndex();
        }
    }
    // Use segment unions on registers to find peak occupancy.
    // Count assigned registers whose segment union covers each point.
    // Optimization: only check at segment start/end boundaries.
    std::vector<std::uint32_t> events;
    for (const auto& lr : ranges) {
        for (const auto& seg : lr.segments) {
            events.push_back(seg.start);
            events.push_back(seg.end);
        }
    }
    std::sort(events.begin(), events.end());
    events.erase(std::unique(events.begin(), events.end()), events.end());

    for (auto idx : events) {
        std::uint32_t live = 0;
        for (const auto& lr : ranges) {
            if (lr.liveAt(idx) && result.reg.count(lr.id)) {
                ++live;
            }
        }
        if (live > result.peak_live) {
            result.peak_live = live;
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
                     const std::vector<LiveRange>& ranges,
                     const Assignment& assignment) {
    for (const auto& lr : ranges) {
        if (lr.empty()) {
            continue;
        }
        os << "  %" << lr.id << ": ";
        for (std::size_t i = 0; i < lr.segments.size(); ++i) {
            if (i > 0) os << " ";
            os << "[" << lr.segments[i].start << ", " << lr.segments[i].end << "]";
        }
        os << " -> ";
        auto it = assignment.reg.find(lr.id);
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

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_ir.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_set>
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
            auto region_begin = 2 * index;  // early slot of first op in region
            trace_ir(std::string("ranges enter ") + (op.is_loop ? "loop" : "region") +
                     " begin=" + std::to_string(region_begin));
            compute_live_ranges_impl(op.body->ops(), ranges, index);
            auto region_end = 2 * index;  // half-open: past last late slot
            trace_ir(std::string("ranges exit ") + (op.is_loop ? "loop" : "region") +
                     " begin=" + std::to_string(region_begin) +
                     " end=" + std::to_string(region_end));

            // For ANY region (loop or branch): values defined BEFORE
            // the region but used INSIDE must have their parent-level
            // segment extended to cover through the child's uses.
            // Without this, the parent's segment (just the def point)
            // would be disconnected from the child's segment (the use
            // point), creating a hole in straight-line code where the
            // register could be incorrectly reused.
            //
            // For loops specifically, also extend committed child
            // segments to cover the full loop body (the loop re-reads
            // the value each iteration).
            // All indices are in slot space (2*op_index for reads, 2*op_index+1 for defs).
            // Region is [region_begin, region_end) in slot space.
            for (auto& [vid, seg] : local) {
                if (seg.first < region_begin) {
                    for (const auto& committed : ranges[vid].segments) {
                        if (committed.start >= region_begin &&
                            committed.start < region_end) {
                            // For loops, extend to cover the full body.
                            // For branches, extend to cover through the child's end.
                            // committed.end is half-open; subtract 1 for local
                            // tracking (flushed as +1 later).
                            auto extend_to = op.is_loop ? region_end - 1
                                                        : committed.end - 1;
                            seg.last = std::max(seg.last, extend_to);
                            break;
                        }
                    }
                }
            }

            if (op.is_loop) {
                // Loop: also extend committed child segments to cover
                // the full loop body.
                for (auto& lr : ranges) {
                    if (lr.empty()) continue;
                    if (lr.beginIndex() >= region_begin) continue;
                    for (auto& seg : lr.segments) {
                        // Half-open: seg.end is one past last use.
                        // Extend if it ends inside the region but before the end.
                        if (seg.end > region_begin && seg.end < region_end) {
                            trace_ir("extend %" + std::to_string(lr.id) +
                                     " loop_end " + std::to_string(seg.end) +
                                     " -> " + std::to_string(region_end));
                            seg.end = region_end;
                        }
                    }
                }
            }
        } else {
            // LLVM-style early/late slot model:
            //   early = 2*index     (reads happen here)
            //   late  = 2*index + 1 (defs happen here)
            // A read extends the range to cover the early slot.
            // A def starts at the late slot. With half-open intervals,
            // [.., early+1) and [late, ..) don't overlap when early+1 == late,
            // enabling coalescing for tied operands and copies.
            auto early = 2 * index;
            auto late  = 2 * index + 1;
            trace_ir("visit op@" + std::to_string(index) +
                     " def=" + (op.def == invalid_value ? std::string("-") : "%" + std::to_string(op.def)) +
                     " reads=" + format_reads(op.reads) +
                     (op.name[0] ? " name=" + std::string(op.name) : ""));
            for (value_id read : op.reads) {
                auto it = local.find(read);
                if (it != local.end()) {
                    if (early > it->second.last) {
                        trace_ir("  last_use %" + std::to_string(read) +
                                 " " + std::to_string(it->second.last) +
                                 " -> " + std::to_string(early));
                        it->second.last = early;
                    }
                } else {
                    trace_ir("  first_use %" + std::to_string(read) +
                             " at " + std::to_string(early));
                    local[read] = {early, early};
                }
            }
            if (op.def != invalid_value) {
                local[op.def] = {late, late};
                if (op.is_copy && op.reads.size() == 1) {
                    ranges[op.def].copy_of = op.reads[0];
                } else if (op.tied_to >= 0 && op.tied_to < static_cast<int>(op.reads.size())) {
                    ranges[op.def].copy_of = op.reads[op.tied_to];
                }
            }
            ++index;
        }
    }

    // Flush local segments into LiveRanges (half-open: end = last + 1).
    for (const auto& [vid, seg] : local) {
        ranges[vid].addSegment({seg.first, seg.last + 1});
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
    return index < it->end;  // half-open: [start, end)
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

    // Half-open: merge if overlapping or adjacent ([0,2) + [2,4) → [0,4)).
    if (it != segments.begin()) {
        auto prev = std::prev(it);
        if (prev->end >= it->start) {
            prev->end = std::max(prev->end, it->end);
            it = segments.erase(it);
            it = prev;
        }
    }

    // Merge with successors if overlapping or adjacent.
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

// Remat all uses of `vid` in the op tree: for each op that reads `vid`,
// insert a clone op (with the same emit closure) immediately before it
// and rewrite that op's reads to the clone's value_id.
// Returns true if the IR was actually modified.
bool remat_all_uses_impl(std::list<Op>& ops, IR& ir, value_id vid, const Op& victim_def) {
    bool modified = false;
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        if (it->body) {
            modified |= remat_all_uses_impl(it->body->ops(), ir, vid, victim_def);
            continue;
        }
        bool reads_victim = false;
        for (auto read : it->reads) {
            if (read == vid) { reads_victim = true; break; }
        }
        if (!reads_victim) continue;

        auto new_id = static_cast<value_id>(ir.value_count());
        ir.set_value_count(new_id + 1);

        Op clone;
        clone.reads = victim_def.reads;  // preserve input dependencies
        clone.def = new_id;
        clone.emit = victim_def.emit;
        clone.name = "remat";
        ops.insert(it, std::move(clone));

        for (auto& read : it->reads) {
            if (read == vid) read = new_id;
        }
        modified = true;
    }
    return modified;
}

// ── Loop unrolling pass ──────────────────────────────────────────────

// Clone a range of ops [begin, end) from a body, remapping value_ids.
// `remap` maps old value_id → new value_id. Values not in the map
// (defined outside the body) are kept as-is. Fresh value_ids are
// allocated from `ir`.
void clone_ops(const std::list<Op>& src_ops,
               std::list<Op>& dst_ops,
               std::list<Op>::const_iterator begin,
               std::list<Op>::const_iterator end,
               IR& ir,
               std::unordered_map<value_id, value_id>& remap) {
    for (auto it = begin; it != end; ++it) {
        const auto& op = *it;
        Op clone;
        // Remap reads
        clone.reads.reserve(op.reads.size());
        for (auto r : op.reads) {
            auto rit = remap.find(r);
            clone.reads.push_back(rit != remap.end() ? rit->second : r);
        }
        // Allocate fresh def if the op defines a value
        if (op.def != invalid_value) {
            value_id new_id = ir.value_count();
            ir.set_value_count(new_id + 1);
            remap[op.def] = new_id;
            clone.def = new_id;
        }
        clone.emit = op.emit;  // share the emit closure
        clone.is_copy = op.is_copy;
        clone.tied_to = op.tied_to;
        clone.is_loop = op.is_loop;
        clone.name = op.name;
        // Note: nested body (regions) not cloned — unrolling only applies
        // to flat loop bodies, not nested structures.
        dst_ops.push_back(std::move(clone));
    }
}

// Estimate peak register pressure in one loop iteration.
// Walks the body ops, tracking live value count at each point.
std::uint32_t estimate_body_pressure(const std::list<Op>& ops) {
    // Track which values are live (defined but not yet last-used).
    // For each value, find its last use index, then count live at each op.
    std::unordered_map<value_id, std::uint32_t> last_use;
    std::uint32_t idx = 0;
    for (const auto& op : ops) {
        for (auto r : op.reads) {
            last_use[r] = idx;
        }
        ++idx;
    }

    std::unordered_set<value_id> live;
    std::uint32_t peak = 0;
    idx = 0;
    for (const auto& op : ops) {
        if (op.def != invalid_value) {
            live.insert(op.def);
        }
        peak = std::max(peak, static_cast<std::uint32_t>(live.size()));
        // Expire values whose last use is this op.
        for (auto r : op.reads) {
            if (last_use[r] == idx) {
                live.erase(r);
            }
        }
        ++idx;
    }
    return peak;
}

// Unroll a single loop body by factor K. Clones all body ops (except
// the loop_footer) K-1 times before the footer.
bool unroll_loop_body(Op& loop_op, IR& ir, std::uint32_t factor) {
    if (factor <= 1 || !loop_op.body) return false;

    auto& body_ops = loop_op.body->ops();
    if (body_ops.empty()) return false;

    // Find the loop_footer — it's the last op with name "loop_footer".
    auto footer_it = body_ops.end();
    for (auto it = body_ops.begin(); it != body_ops.end(); ++it) {
        if (std::string(it->name) == "loop_footer") {
            footer_it = it;
        }
    }

    // Clone everything before the footer, K-1 times, inserting before footer.
    for (std::uint32_t k = 1; k < factor; ++k) {
        std::unordered_map<value_id, value_id> remap;
        clone_ops(body_ops, body_ops, body_ops.begin(), footer_it, ir, remap);
    }

    trace_ir("unroll loop by " + std::to_string(factor) +
             " (" + std::to_string(body_ops.size()) + " ops in body)");
    return true;
}

}  // namespace

// ── Public unroll_loops pass ────────────────────────────────────────

bool unroll_loops(IR& ir, std::uint32_t pool_size, UnrollStrategy strategy) {
    if (strategy == UnrollStrategy::none) return false;

    bool modified = false;

    for (auto& op : ir.ops()) {
        if (!op.body || !op.is_loop) continue;

        auto& body_ops = op.body->ops();
        auto pressure = estimate_body_pressure(body_ops);
        trace_ir("unroll: found loop, body_ops=" + std::to_string(body_ops.size()) +
                 " pressure=" + std::to_string(pressure) +
                 " pool=" + std::to_string(pool_size));
        if (pressure == 0) continue;

        std::uint32_t factor = 1;

        if (strategy == UnrollStrategy::heuristic) {
            // LLVM-style: unroll as much as register pressure allows.
            factor = pool_size / pressure;
            if (factor < 2) continue;
            // Cap at 8 to avoid code bloat.
            factor = std::min(factor, 8u);
        } else if (strategy == UnrollStrategy::feedback) {
            // Feedback-directed: try increasing factors until allocation fails.
            for (std::uint32_t try_factor = pool_size / pressure;
                 try_factor >= 2; try_factor /= 2) {
                // Build a trial IR copy — expensive but optimal.
                // For now, use the heuristic as a starting point.
                // @todo claude: implement trial allocation for feedback mode
                factor = try_factor;
                break;
            }
            if (factor < 2) continue;
            factor = std::min(factor, 8u);
        }

        if (unroll_loop_body(op, ir, factor)) {
            modified = true;
        }
    }

    return modified;
}

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

std::optional<Assignment> linear_scan(IR& ir,
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
    // Half-open intervals: [a, b) and [c, d) overlap iff a < d && c < b.
    auto interferes = [&](const LiveRange& lr, std::uint32_t r) -> bool {
        for (const auto& s : lr.segments) {
            for (const auto& e : reg_segments[r]) {
                if (s.start < e.end && e.start < s.end) {
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
            // All registers interfere. Try to remat a victim: find the
            // rematerializable value (empty reads) with the longest range
            // among all already-assigned values. Remat replaces all its
            // uses with local clones, making the original dead and freeing
            // its register pressure.
            std::unordered_map<value_id, Op*> def_map;
            collect_def_ops(ir.ops(), def_map);

            // Only consider victims whose segments INTERFERE with the
            // failing value — only those contribute to the pressure at
            // this specific point. This prevents infinite remat loops
            // where short-lived clones far from the pressure point keep
            // getting picked without reducing peak pressure.
            auto interferes_with_lr = [&](const LiveRange& victim_lr) -> bool {
                for (const auto& vs : victim_lr.segments) {
                    for (const auto& ls : lr.segments) {
                        if (vs.start < ls.end && ls.start < vs.end)
                            return true;
                    }
                }
                return false;
            };

            // A value is rematerializable if:
            //  (a) its def op has no reads (constant/broadcast), OR
            //  (b) all of its def op's inputs have ranges that span the
            //      victim's entire lifetime — so the clone's reads are
            //      already live at every use site, adding no pressure.
            auto is_rematerializable = [&](value_id v) -> bool {
                auto dit = def_map.find(v);
                if (dit == def_map.end()) return false;
                const auto& reads = dit->second->reads;
                if (reads.empty()) return true;
                for (auto input : reads) {
                    if (input >= ranges.size()) return false;
                    if (ranges[input].beginIndex() > ranges[v].beginIndex()) return false;
                    if (ranges[input].endIndex() < ranges[v].endIndex()) return false;
                }
                return true;
            };

            value_id best_victim = invalid_value;
            std::uint32_t best_range = 0;
            for (const auto& [v, _reg] : result.reg) {
                if (!is_rematerializable(v)) continue;
                if (!interferes_with_lr(ranges[v])) continue;
                auto range = ranges[v].endIndex() - ranges[v].beginIndex();
                if (range > best_range) {
                    best_range = range;
                    best_victim = v;
                }
            }
            // Also consider the current unassigned value.
            if (is_rematerializable(vid)) {
                auto range = lr.endIndex() - lr.beginIndex();
                if (range > best_range) {
                    best_victim = vid;
                }
            }

            if (best_victim != invalid_value) {
                auto* vdef = def_map[best_victim];
                if (remat_all_uses_impl(ir.ops(), ir, best_victim, *vdef)) {
                    trace_ir("remat %" + std::to_string(best_victim) +
                             " (all uses) — retry allocation");
                    return std::nullopt;  // IR modified, caller retries
                }
            }

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

bool unit_test_api_remat_value(IR& ir, value_id vid) {
    std::unordered_map<value_id, Op*> def_map;
    collect_def_ops(ir.ops(), def_map);
    auto it = def_map.find(vid);
    if (it == def_map.end()) return false;
    return remat_all_uses_impl(ir.ops(), ir, vid, *it->second);
}

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

// ── Pass implementations ────────────────────────────────────────────

bool LiveRangeAnalysis::run(IR& ir, PassContext& ctx) {
    ctx.ranges = compute_live_ranges(ir);
    return false;  // analysis, no IR modification
}

bool LoopUnrollPass::run(IR& ir, PassContext& ctx) {
    return unroll_loops(ir, ctx.pool_size, strategy);
}

bool RegisterAllocator::run(IR& ir, PassContext& ctx) {
    // Remat loop: linear_scan may modify the IR and return nullopt,
    // requiring recomputation of live ranges.
    for (std::size_t attempt = 0, limit = ctx.ranges.size();
         attempt < limit; ++attempt) {
        auto result = linear_scan(ir, ctx.ranges, ctx.pool_size);
        if (result) {
            ctx.assignment = std::move(*result);
            return false;
        }
        // Remat modified the IR — recompute live ranges and retry.
        ctx.ranges = compute_live_ranges(ir);
    }
    OPENVINO_THROW("RegisterAllocator: allocation failed after remat exhaustion");
}

bool DumpPass::run(IR& ir, PassContext& ctx) {
    if (!ctx.dump) return false;
    std::ostringstream os;
    os << "=== " << label << ": pool_size=" << ctx.pool_size
       << " values=" << ctx.ranges.size() << " ===\n";
    dump_ops(os, ir);
    if (ctx.assignment) {
        os << "--- assignment ---\n";
        dump_assignment(os, ctx.ranges, *ctx.assignment);
    } else {
        os << "--- live ranges ---\n";
        dump_assignment(os, ctx.ranges, Assignment{});
    }
    std::cout << os.str();
    return false;
}

bool LoweringPass::run(IR& ir, PassContext& ctx) {
    OPENVINO_ASSERT(ctx.assignment, "LoweringPass: no assignment available");
    OPENVINO_ASSERT(ctx.lower_fn, "LoweringPass: no lowering callback set");
    ctx.lower_fn(ir, *ctx.assignment);
    return false;
}

PassManager build_default_pipeline(UnrollStrategy unroll) {
    PassManager pm;
    pm.add<LoopUnrollPass>(unroll);
    pm.add<LiveRangeAnalysis>();
    pm.add<DumpPass>("IR before allocation");
    pm.add<RegisterAllocator>();
    pm.add<DumpPass>("IR after allocation");
    pm.add<LoweringPass>();
    return pm;
}

}  // namespace ov::intel_cpu::jit_kernel_ir

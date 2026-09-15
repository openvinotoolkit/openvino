// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "jit_kernel_ir.hpp"

#include "openvino/core/except.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ov::intel_cpu::jit_kernel_ir {

namespace {

bool env_enabled(const char* name) {
    return std::getenv(name) != nullptr;
}

struct debug_config {
    bool trace_all = env_enabled("OV_JIT_IR_TRACE");
};

const debug_config& get_debug_config() {
    static const debug_config cfg{};
    return cfg;
}

void trace_ir(const std::string& msg) {
    if (!get_debug_config().trace_all) {
        return;
    }
    std::cout << "[jit_kernel_ir] " << msg << "\n";
}

// ── CFG over the structured region tree ────────────────────────────────
//
// The IR nests bodies instead of holding a block graph, so liveness starts
// by materializing the control flow that the region ops' emit closures
// actually implement:
//
//   loop region:        preheader -> header
//                       header    -> {body entry, loop exit}
//                       latch     -> header                (back edge)
//   conditional region: header    -> {body entry, join}     (body may be skipped)
//                       body exit -> join
//
// A loop gets its own header block because the back edge targets the
// compare, not the code that initialised the counter — folding the two
// together would make values that live across the loop appear dead inside
// it.
//
// Every op owns one index, region ops included: a region op is the branch
// instruction of its header block, so its reads (loop counter, bound) are
// live exactly where the compare reads them.

struct BlockOp {
    const Op* op = nullptr;
    std::uint32_t index = 0;   // early slot = 2*index, late slot = 2*index+1
};

struct Block {
    std::vector<BlockOp> ops;
    std::vector<std::uint32_t> succs;
    std::unordered_set<value_id> uses;      // read before any def in this block
    std::unordered_set<value_id> defs;
    std::unordered_set<value_id> live_in;
    std::unordered_set<value_id> live_out;
};

class CFG {
public:
    explicit CFG(const IR& ir) {
        const auto entry = add_block();
        build(ir.ops(), entry);
        compute_local_sets();
        solve();
        trace();
    }

    [[nodiscard]] const std::vector<Block>& blocks() const noexcept { return _blocks; }

private:
    std::uint32_t add_block() {
        _blocks.emplace_back();
        return static_cast<std::uint32_t>(_blocks.size() - 1);
    }

    // Appends `ops` starting in block `cur`; returns the block that control
    // reaches after the list. Blocks are created in walk order, so block
    // order is ascending slot order and consecutive blocks are adjacent in
    // slot space (which lets addSegment merge across block boundaries).
    std::uint32_t build(const std::list<Op>& ops, std::uint32_t cur) {
        for (const auto& op : ops) {
            if (!op.body) {
                _blocks[cur].ops.push_back({&op, _index++});
                continue;
            }

            if (op.is_loop) {
                const auto header = add_block();
                _blocks[cur].succs.push_back(header);
                _blocks[header].ops.push_back({&op, _index++});

                const auto body_entry = add_block();
                _blocks[header].succs.push_back(body_entry);
                const auto latch = build(op.body->ops(), body_entry);
                _blocks[latch].succs.push_back(header);

                const auto loop_exit = add_block();
                _blocks[header].succs.push_back(loop_exit);
                cur = loop_exit;
            } else {
                _blocks[cur].ops.push_back({&op, _index++});

                const auto body_entry = add_block();
                _blocks[cur].succs.push_back(body_entry);
                const auto body_exit = build(op.body->ops(), body_entry);

                const auto join = add_block();
                _blocks[cur].succs.push_back(join);
                _blocks[body_exit].succs.push_back(join);
                cur = join;
            }
        }
        return cur;
    }

    void compute_local_sets() {
        for (auto& b : _blocks) {
            for (const auto& bop : b.ops) {
                // Reads happen at the early slot, the def at the late slot,
                // so a read of the value the op redefines is still a use.
                for (auto r : bop.op->reads) {
                    if (b.defs.count(r) == 0) {
                        b.uses.insert(r);
                    }
                }
                if (bop.op->def != invalid_value) {
                    b.defs.insert(bop.op->def);
                }
            }
        }
    }

    // Backward dataflow to fixpoint:
    //   live_out[B] = U live_in[S], S in succ(B)
    //   live_in[B]  = uses[B] U (live_out[B] - defs[B])
    void solve() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto i = _blocks.size(); i-- > 0;) {
                std::unordered_set<value_id> out;
                for (auto s : _blocks[i].succs) {
                    out.insert(_blocks[s].live_in.begin(), _blocks[s].live_in.end());
                }
                auto in = _blocks[i].uses;
                for (auto v : out) {
                    if (_blocks[i].defs.count(v) == 0) {
                        in.insert(v);
                    }
                }
                if (out != _blocks[i].live_out || in != _blocks[i].live_in) {
                    _blocks[i].live_out = std::move(out);
                    _blocks[i].live_in = std::move(in);
                    changed = true;
                }
            }
        }
    }

    void trace() const {
        if (!get_debug_config().trace_all) {
            return;
        }
        for (std::size_t i = 0; i < _blocks.size(); ++i) {
            const auto& b = _blocks[i];
            std::string msg = "bb" + std::to_string(i) + " ops=" + std::to_string(b.ops.size());
            if (!b.ops.empty()) {
                msg += " slots=[" + std::to_string(2 * b.ops.front().index) + ", " +
                       std::to_string(2 * b.ops.back().index + 2) + ")";
            }
            msg += " succs={";
            for (std::size_t s = 0; s < b.succs.size(); ++s) {
                if (s != 0) {
                    msg += ", ";
                }
                msg += "bb" + std::to_string(b.succs[s]);
            }
            msg += "} live_in=" + std::to_string(b.live_in.size()) +
                   " live_out=" + std::to_string(b.live_out.size());
            trace_ir(msg);
        }
    }

    std::vector<Block> _blocks;
    std::uint32_t _index = 0;
};

// Value numbers: one per def, in slot order. Also propagates register
// class and the coalescing hint from the defining op.
void collect_value_numbers(const std::vector<Block>& blocks, std::vector<LiveRange>& ranges) {
    for (const auto& b : blocks) {
        for (const auto& bop : b.ops) {
            const auto d = bop.op->def;
            if (d == invalid_value) {
                continue;
            }
            auto& lr = ranges[d];
            lr.rc = bop.op->def_rc;
            // Early-clobber defs start at the early slot, so they overlap
            // the op's own reads and cannot be assigned the same register.
            lr.getNextValue(2 * bop.index + (bop.op->early_clobber ? 0 : 1));
            if (bop.op->is_copy && !bop.op->reads.empty()) {
                lr.copy_of = bop.op->reads[0];
            } else if (bop.op->tied_to >= 0 &&
                       bop.op->tied_to < static_cast<int>(bop.op->reads.size())) {
                lr.copy_of = bop.op->reads[static_cast<std::size_t>(bop.op->tied_to)];
            }
        }
    }
}

// Turns block-level liveness into segments. Per block: values live-in open
// at the block start, defs open at their late slot, and every open segment
// closes at the block end (live-out) or at the last read inside the block.
// Consecutive blocks are adjacent in slot space, so addSegment merges
// pass-through liveness into one segment per value per live region.
void build_segments(const std::vector<Block>& blocks, std::vector<LiveRange>& ranges) {
    auto valno_at = [](const LiveRange& lr, std::uint32_t slot) -> std::uint32_t {
        // The latest def at or before `slot`. Values whose def lives in
        // another block keep valno 0 — there is no PHI numbering, and
        // nothing but the dump consumes value numbers today.
        std::uint32_t best = 0;
        for (const auto& vn : lr.valnos) {
            if (vn.def <= slot) {
                best = vn.id;
            }
        }
        return best;
    };

    for (const auto& b : blocks) {
        if (b.ops.empty()) {
            continue;  // pass-through block; liveness already flows through it
        }
        const auto block_begin = 2 * b.ops.front().index;
        const auto block_end = 2 * b.ops.back().index + 2;  // half-open

        std::unordered_map<value_id, std::uint32_t> open;       // value -> segment start
        std::unordered_map<value_id, std::uint32_t> last_read;   // value -> early slot

        for (auto v : b.live_in) {
            open[v] = block_begin;
        }

        for (const auto& bop : b.ops) {
            const auto early = 2 * bop.index;
            const auto late = early + 1;

            for (auto r : bop.op->reads) {
                if (open.find(r) == open.end()) {
                    open[r] = early;  // defensive: read with no live-in and no def
                }
                last_read[r] = early;
            }

            const auto d = bop.op->def;
            if (d == invalid_value) {
                continue;
            }
            const auto def_slot = bop.op->early_clobber ? early : late;
            auto it = open.find(d);
            if (it != open.end()) {
                // Re-def inside the block (post-TwoAddressPass): close the
                // previous segment before starting the new one.
                const auto lr_it = last_read.find(d);
                const auto end = lr_it != last_read.end() ? lr_it->second + 1
                                                          : it->second + 1;
                ranges[d].addSegment({it->second, std::max(end, it->second + 1),
                                      valno_at(ranges[d], it->second)});
                last_read.erase(d);
            }
            open[d] = def_slot;
        }

        for (const auto& [v, start] : open) {
            std::uint32_t end = 0;
            if (b.live_out.count(v) != 0) {
                end = block_end;
            } else {
                auto it = last_read.find(v);
                end = it != last_read.end() ? it->second + 1 : start + 1;
            }
            ranges[v].addSegment({start, std::max(end, start + 1), valno_at(ranges[v], start)});
        }
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
    return it->contains(index);
}

bool LiveRange::overlaps(const LiveRange& other) const noexcept {
    // Two-pointer merge scan — both segment lists are sorted by start.
    auto i = segments.begin(), ie = segments.end();
    auto j = other.segments.begin(), je = other.segments.end();
    while (i != ie && j != je) {
        if (i->overlaps(*j)) return true;
        if (i->start < j->start) ++i; else ++j;
    }
    return false;
}

bool LiveRange::overlaps(std::uint32_t start, std::uint32_t end) const noexcept {
    for (const auto& s : segments) {
        if (s.overlaps(start, end)) return true;
        if (s.start >= end) break;  // remaining segments are past the query
    }
    return false;
}

std::uint32_t LiveRange::getNextValue(std::uint32_t def) {
    auto idx = static_cast<std::uint32_t>(valnos.size());
    valnos.push_back(VNInfo{idx, def});
    return idx;
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

    const CFG cfg(ir);
    collect_value_numbers(cfg.blocks(), ranges);
    build_segments(cfg.blocks(), ranges);
    return ranges;
}

namespace {

// Maps value_id -> defining Op by walking the region tree. Rematerializability
// is derived from the def op itself (no reads, or reads that outlive the
// victim), so no separate bookkeeping is needed.
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
        clone.def_rc = victim_def.def_rc;  // preserve register class
        clone.early_clobber = victim_def.early_clobber;
        clone.may_load = victim_def.may_load;
        clone.mem_ptr_read = victim_def.mem_ptr_read;
        clone.mem_offset = victim_def.mem_offset;
        clone.name = "remat";
        ops.insert(it, std::move(clone));

        for (auto& read : it->reads) {
            if (read == vid) read = new_id;
        }
        modified = true;
    }
    return modified;
}

}  // namespace

std::optional<Assignment> assign_registers(IR& ir,
                                           std::vector<LiveRange>& ranges,
                                           PassContext& ctx) {
    // Per-register interference allocation across register files. Each
    // class has an allocation order — the physical registers it may use, in
    // preference order — mirroring LLVM's AllocationOrder
    // (RegisterClassInfo::getOrder). Files never interfere with each other.
    auto pool_for = [&ctx](RegisterClass rc) -> const std::vector<std::uint32_t>& {
        return ctx.pool(rc);
    };

    // Visit live ranges in order of first definition. Single unified work
    // list across register classes — the pools are what differ.
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

    // Segment unions per pool slot, one array per class.
    std::array<std::vector<std::vector<Segment>>, register_class_count> reg_segments;
    for (std::size_t rc = 0; rc < register_class_count; ++rc) {
        reg_segments[rc].resize(pool_for(static_cast<RegisterClass>(rc)).size());
    }

    auto segs_for = [&](RegisterClass rc, std::uint32_t slot) -> std::vector<Segment>& {
        return reg_segments[static_cast<std::size_t>(rc)][slot];
    };

    auto pool_size_for = [&](RegisterClass rc) -> std::uint32_t {
        return static_cast<std::uint32_t>(pool_for(rc).size());
    };

    // Pool slot → physical register index.
    auto phys_idx = [&](RegisterClass rc, std::uint32_t slot) -> std::uint32_t {
        return pool_for(rc)[slot];
    };

    // Physical register index → pool slot (for coalescing lookups).
    auto slot_of = [&](RegisterClass rc, std::uint32_t phys) -> std::uint32_t {
        const auto& pool = pool_for(rc);
        for (std::uint32_t i = 0; i < pool.size(); ++i) {
            if (pool[i] == phys) {
                return i;
            }
        }
        return static_cast<std::uint32_t>(pool.size());  // not found
    };

    // Check if any segment of `lr` overlaps any segment on pool slot `slot`.
    auto interferes = [&](const LiveRange& lr, std::uint32_t slot) -> bool {
        auto& seg_union = segs_for(lr.rc, slot);
        for (const auto& s : lr.segments) {
            for (const auto& e : seg_union) {
                if (s.overlaps(e)) return true;
            }
        }
        return false;
    };

    // Add all segments of `lr` to pool slot `slot`.
    auto add_to_reg = [&](const LiveRange& lr, std::uint32_t slot) {
        auto& seg_union = segs_for(lr.rc, slot);
        for (const auto& s : lr.segments) {
            seg_union.push_back(s);
        }
    };

    Assignment result;
    result.reg.reserve(order.size());

    for (auto lr_idx : order) {
        const auto& lr = ranges[lr_idx];
        auto vid = lr.id;
        auto rc = lr.rc;
        auto ps = pool_size_for(rc);
        const char* rc_tag = to_string(rc);

        trace_ir("alloc %" + std::to_string(vid) + " " + rc_tag +
                 " segs=" + std::to_string(lr.segments.size()) +
                 " [" + std::to_string(lr.beginIndex()) +
                 "," + std::to_string(lr.endIndex()) + ")");

        if (ps == 0) {
            throw allocation_failure(
                "jit_kernel_ir::assign_registers: no " + std::string(rc_tag) +
                " registers available for value %" + std::to_string(vid) +
                " (the ISA has no such register file, or the pool is empty)");
        }

        // Trivial coalescing: prefer the source's register if compatible.
        // Only coalesce within the same register class.
        if (lr.copy_of != invalid_value) {
            auto src_it = result.reg.find(lr.copy_of);
            if (src_it != result.reg.end()) {
                auto src_rc = (lr.copy_of < ranges.size()) ? ranges[lr.copy_of].rc : rc;
                if (src_rc == rc) {
                    auto src_slot = slot_of(rc, src_it->second.idx);
                    if (src_slot < ps && !interferes(lr, src_slot)) {
                        trace_ir("  coalesce %" + std::to_string(vid) +
                                 " with %" + std::to_string(lr.copy_of) +
                                 " on p" + std::to_string(src_it->second.idx));
                        result.reg.emplace(vid, src_it->second);
                        add_to_reg(lr, src_slot);
                        continue;
                    }
                }
            }
        }

        // Find first non-interfering register in this class's pool.
        bool assigned = false;
        for (std::uint32_t slot = 0; slot < ps; ++slot) {
            if (!interferes(lr, slot)) {
                auto pidx = phys_idx(rc, slot);
                trace_ir("  assign %" + std::to_string(vid) +
                         " -> p" + std::to_string(pidx));
                result.reg.emplace(vid, PhysReg{pidx});
                add_to_reg(lr, slot);
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            // All registers interfere. Try to remat a victim.
            std::unordered_map<value_id, Op*> def_map;
            collect_def_ops(ir.ops(), def_map);

            auto interferes_with_lr = [&](const LiveRange& victim_lr) -> bool {
                return victim_lr.overlaps(lr);
            };

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

            // Only remat victims of the same register class.
            value_id best_victim = invalid_value;
            std::uint32_t best_range = 0;
            for (const auto& [v, _reg] : result.reg) {
                if (v < ranges.size() && ranges[v].rc != rc) continue;
                if (!is_rematerializable(v)) continue;
                if (!interferes_with_lr(ranges[v])) continue;
                auto range = ranges[v].endIndex() - ranges[v].beginIndex();
                if (range > best_range) {
                    best_range = range;
                    best_victim = v;
                }
            }
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
                    return std::nullopt;
                }
            }

            // No remat possible and there is no spiller yet: fail loudly.
            // A spiller needs (a) stack slots sized after allocation, which
            // means frame setup must move behind the allocator, and (b)
            // target hooks to emit store/load of a physical register, which
            // the IR layer does not own. Until both exist, reporting the
            // failure is the only correct behaviour — silently dropping the
            // spill code would miscompile.
            std::uint32_t interfering = 0;
            for (const auto& [v, _reg] : result.reg) {
                if (v < ranges.size() && ranges[v].rc == rc && ranges[v].overlaps(lr)) {
                    ++interfering;
                }
            }
            throw allocation_failure(
                "jit_kernel_ir::assign_registers: " + std::string(rc_tag) + " pool of " +
                std::to_string(ps) + " registers exhausted for value %" +
                std::to_string(vid) + " at slot " + std::to_string(lr.beginIndex()) +
                " (" + std::to_string(interfering) + " interfering live values, none"
                " rematerializable; no spiller implemented)");
        }
    }

    // Compute peak_live (vec only — for diagnostics).
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
            // Region ops own an index like any other op — they emit the
            // compare/branch — so the dump numbering matches slot indices.
            os << indent << i << ": " << (op.is_loop ? "LOOP" : "REGION") << "(";
            for (std::size_t r = 0; r < op.reads.size(); ++r) {
                if (r != 0) os << ", ";
                os << "%" << op.reads[r];
            }
            os << ") {\n";
            ++i;
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

// ── Verifier ────────────────────────────────────────────────────────────

namespace {

// Collects, for every op in the tree, the values it defines and reads,
// so the verifier can check completeness without re-walking per value.
struct OpRefs {
    std::unordered_set<value_id> defined;     // has a defining op
    std::vector<value_id> referenced;         // appears as read or def
    std::string shape_error;                  // first structural error found
};

void collect_op_refs(const std::list<Op>& ops, OpRefs& refs) {
    for (const auto& op : ops) {
        if (op.def != invalid_value) {
            refs.defined.insert(op.def);
            refs.referenced.push_back(op.def);
        }
        for (auto r : op.reads) {
            refs.referenced.push_back(r);
        }
        if (refs.shape_error.empty()) {
            if (op.tied_to >= 0) {
                refs.shape_error = "op '" + std::string(op.name) +
                    "' still carries tied_to=" + std::to_string(op.tied_to) +
                    " after TwoAddressPass";
            } else if (op.is_copy && op.reads.size() != 1) {
                refs.shape_error = "copy op '" + std::string(op.name) +
                    "' has " + std::to_string(op.reads.size()) + " reads, expected 1";
            }
        }
        if (op.body) {
            collect_op_refs(op.body->ops(), refs);
        }
    }
}

}  // namespace

void verify(const IR& ir,
            const std::vector<LiveRange>& ranges,
            const Assignment& assignment,
            const PassContext& ctx) {
    auto fail = [](const std::string& what) {
        throw verification_failure("jit_kernel_ir::verify: " + what);
    };

    OpRefs refs;
    collect_op_refs(ir.ops(), refs);
    if (!refs.shape_error.empty()) {
        fail(refs.shape_error);
    }

    // 1. Live range structure and register class agreement.
    for (const auto& lr : ranges) {
        std::uint32_t prev_end = 0;
        for (std::size_t i = 0; i < lr.segments.size(); ++i) {
            const auto& s = lr.segments[i];
            if (s.start >= s.end) {
                fail("value %" + std::to_string(lr.id) + " has empty segment [" +
                     std::to_string(s.start) + ", " + std::to_string(s.end) + ")");
            }
            if (i > 0 && s.start < prev_end) {
                fail("value %" + std::to_string(lr.id) +
                     " segments are unsorted or overlapping around index " +
                     std::to_string(s.start));
            }
            if (s.valno >= lr.valnos.size()) {
                fail("value %" + std::to_string(lr.id) + " segment references valno " +
                     std::to_string(s.valno) + " but only " +
                     std::to_string(lr.valnos.size()) + " exist");
            }
            prev_end = s.end;
        }
    }

    // 2. No undefined reads. A read of a value nothing defines means an IR
    //    rewrite (remat, unroll, two-address) left a dangling id behind.
    for (auto v : refs.referenced) {
        if (refs.defined.count(v) == 0) {
            fail("value %" + std::to_string(v) + " is read but never defined");
        }
    }

    // 3. Completeness: lowering resolves every read and def through the
    //    assignment, so anything referenced must be assigned.
    for (auto v : refs.referenced) {
        if (assignment.reg.count(v) == 0) {
            fail("value %" + std::to_string(v) + " is referenced by an op but "
                 "has no register assigned");
        }
    }

    // 5. Assigned register must belong to the value's pool.
    for (const auto& [v, preg] : assignment.reg) {
        if (v >= ranges.size()) {
            fail("assignment mentions unknown value %" + std::to_string(v));
        }
        const auto rc = ranges[v].rc;
        const auto& pool = ctx.pool(rc);
        if (std::find(pool.begin(), pool.end(), preg.idx) == pool.end()) {
            fail("value %" + std::to_string(v) + " assigned " + to_string(rc) + " " +
                 std::to_string(preg.idx) + " which is not in its allocable pool");
        }
    }

    // 4. Interference: values sharing a physical register (within a class)
    //    must have disjoint live ranges. Group by (class, phys) first so the
    //    pairwise check stays small.
    std::map<std::pair<std::uint8_t, std::uint32_t>, std::vector<value_id>> by_reg;
    for (const auto& [v, preg] : assignment.reg) {
        if (v >= ranges.size() || ranges[v].empty()) {
            continue;
        }
        by_reg[{static_cast<std::uint8_t>(ranges[v].rc), preg.idx}].push_back(v);
    }
    for (const auto& [key, values] : by_reg) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            for (std::size_t j = i + 1; j < values.size(); ++j) {
                const auto& a = ranges[values[i]];
                const auto& b = ranges[values[j]];
                if (a.overlaps(b)) {
                    fail("values %" + std::to_string(a.id) + " and %" +
                         std::to_string(b.id) + " share " +
                         to_string(static_cast<RegisterClass>(key.first)) + " " +
                         std::to_string(key.second) +
                         " but their live ranges overlap");
                }
            }
        }
    }
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

// ── FoldMemoryOperandsPass ──────────────────────────────────────────────

namespace {

// How many times `vid` is read anywhere in the tree, region headers and
// nested bodies included. A load may only be folded away when its value
// has exactly one consumer.
std::uint32_t count_reads(const std::list<Op>& ops, value_id vid) {
    std::uint32_t n = 0;
    for (const auto& op : ops) {
        for (auto r : op.reads) {
            if (r == vid) {
                ++n;
            }
        }
        if (op.body) {
            n += count_reads(op.body->ops(), vid);
        }
    }
    return n;
}

// Folds within one straight-line list. Returns how many folds were made.
std::uint32_t fold_in_list(std::list<Op>& ops, const IR& ir) {
    std::uint32_t folded = 0;

    for (auto load_it = ops.begin(); load_it != ops.end();) {
        auto& load = *load_it;
        const bool foldable_load = load.may_load && !load.may_store &&
                                   load.def != invalid_value && load.mem_ptr_read >= 0 &&
                                   load.folded_read < 0 && load.foldable_reads == 0;
        if (!foldable_load || count_reads(ir.ops(), load.def) != 1) {
            ++load_it;
            continue;
        }

        // Scan forward for the single consumer. Stop at anything that may
        // write memory — the load cannot move past it — at any region,
        // since the consumer must be on this straight-line path, and at any
        // op that touches the base pointer: pointer bumps mutate the
        // register they declare as a read, so an address folded past one
        // would resolve differently.
        const auto base_ptr = load.reads[static_cast<std::size_t>(load.mem_ptr_read)];
        auto use_it = std::next(load_it);
        bool blocked = false;
        for (; use_it != ops.end(); ++use_it) {
            if (use_it->body || use_it->may_store) {
                blocked = true;
                break;
            }
            const auto& reads = use_it->reads;
            const bool uses_value = std::find(reads.begin(), reads.end(), load.def) != reads.end();
            if (uses_value) {
                break;
            }
            if (std::find(reads.begin(), reads.end(), base_ptr) != reads.end()) {
                blocked = true;
                break;
            }
        }
        if (blocked || use_it == ops.end()) {
            ++load_it;
            continue;
        }

        auto& use = *use_it;
        if (use.foldable_reads == 0 || !use.fold_emit || use.folded_read >= 0) {
            ++load_it;
            continue;
        }

        // Which operand of the consumer holds the loaded value, and may
        // that operand come from memory?
        int fold_read = -1;
        for (std::size_t i = 0; i < use.reads.size() && i < 8; ++i) {
            if (use.reads[i] == load.def && ((use.foldable_reads >> i) & 1U) != 0U) {
                fold_read = static_cast<int>(i);
                break;
            }
        }
        if (fold_read < 0) {
            ++load_it;
            continue;
        }

        // Rewrite the consumer: the folded operand now names the base
        // pointer, and the memory form of the instruction takes over.
        use.reads[static_cast<std::size_t>(fold_read)] = base_ptr;
        use.folded_read = fold_read;
        use.mem_offset = load.mem_offset;
        use.may_load = true;
        use.mem_ptr_read = fold_read;
        use.emit = use.fold_emit;

        trace_ir("fold load %" + std::to_string(load.def) + " into " +
                 std::string(use.name) + " operand " + std::to_string(fold_read));

        load_it = ops.erase(load_it);
        ++folded;
    }

    return folded;
}

std::uint32_t fold_recursive(std::list<Op>& ops, const IR& ir) {
    std::uint32_t folded = fold_in_list(ops, ir);
    for (auto& op : ops) {
        if (op.body) {
            folded += fold_recursive(op.body->ops(), ir);
        }
    }
    return folded;
}

}  // namespace

bool FoldMemoryOperandsPass::run(IR& ir, PassContext& ctx) {
    if (ctx.disable_memory_folding) {
        return false;
    }
    return fold_recursive(ir.ops(), ir) != 0;
}

namespace {

using use_counts = std::unordered_map<value_id, std::uint32_t>;

void count_uses(const std::list<Op>& ops, use_counts& uses) {
    for (const auto& op : ops) {
        for (auto read : op.reads) {
            ++uses[read];
        }
        if (op.body) {
            count_uses(op.body->ops(), uses);
        }
    }
}

// MachineInstr::isDead, restricted to what this IR models: the def must be
// unused, and the op must have no effect beyond it. Region ops are the
// branch instructions of their headers (LLVM: terminators) and def-less
// ops are the DSL's side-effecting form, so neither is a candidate.
bool is_dead(const Op& op, const use_counts& uses) {
    if (op.body || op.def == invalid_value || op.may_store) {
        return false;
    }
    auto it = uses.find(op.def);
    return it == uses.end() || it->second == 0;
}

// Backwards, decrementing use counts as ops die, so an op that only fed a
// dead op dies in the same sweep.
std::uint32_t erase_dead(std::list<Op>& ops, use_counts& uses) {
    std::uint32_t erased = 0;
    for (auto it = ops.end(); it != ops.begin();) {
        --it;
        if (it->body) {
            erased += erase_dead(it->body->ops(), uses);
            continue;
        }
        if (!is_dead(*it, uses)) {
            continue;
        }
        for (auto read : it->reads) {
            auto found = uses.find(read);
            if (found != uses.end() && found->second > 0) {
                --found->second;
            }
        }
        // erase() returns the following op; the loop's --it steps back to
        // the one before the erased op, which is where to continue.
        it = ops.erase(it);
        ++erased;
    }
    return erased;
}

}  // namespace

bool DeadDefElimPass::run(IR& ir, PassContext& ctx) {
    use_counts uses;
    count_uses(ir.ops(), uses);

    std::uint32_t erased = erase_dead(ir.ops(), uses);
    if (erased == 0) {
        return false;
    }
    // LLVM re-runs the sweep while anything changed; the backwards walk
    // already catches chains, so this only picks up cross-region leftovers.
    while (erase_dead(ir.ops(), uses) != 0) {
    }
    if (ctx.trace) {
        std::cout << "[jit_ir] DeadDefElim: erased " << erased << " dead def(s)\n";
    }
    return true;
}

// Recursive helper for TwoAddressPass.
static bool two_address_rewrite(std::list<Op>& ops, IR& ir) {
    bool modified = false;
    for (auto it = ops.begin(); it != ops.end(); ++it) {
        // Recurse into nested regions.
        if (it->body) {
            if (two_address_rewrite(it->body->ops(), ir)) {
                modified = true;
            }
            continue;
        }
        if (it->tied_to < 0 || it->tied_to >= static_cast<int>(it->reads.size())) {
            continue;
        }
        // Found a tied-operand op. Insert a COPY before it.
        // LLVM approach: break SSA — the COPY and the FMA both use
        // the FMA's def value_id. The COPY defines it (from the seed),
        // the FMA redefines it (in-place). One live range, one register.
        auto tied_idx = static_cast<std::size_t>(it->tied_to);
        auto seed_vid = it->reads[tied_idx];
        auto def_vid = it->def;  // reuse FMA's def for the COPY

        Op copy_op;
        copy_op.reads = {seed_vid};
        copy_op.def = def_vid;       // same value_id as the FMA's def
        copy_op.is_copy = true;
        copy_op.def_rc = it->def_rc; // inherit register class from the tied op
        copy_op.name = "two_addr_copy";
        copy_op.emit = [](const EmitContext&) {};  // lowering handles is_copy

        // Rewrite the FMA: reads[tied_idx] = def_vid (reads itself).
        // The FMA now reads and writes the same value_id, matching LLVM's
        // post-TwoAddressPass representation.
        it->reads[tied_idx] = def_vid;
        it->tied_to = -1;  // constraint satisfied by construction

        ops.insert(it, std::move(copy_op));
        modified = true;
    }
    return modified;
}

bool TwoAddressPass::run(IR& ir, PassContext& ctx) {
    return two_address_rewrite(ir.ops(), ir);
}

bool LiveRangeAnalysis::run(IR& ir, PassContext& ctx) {
    ctx.ranges = compute_live_ranges(ir);
    return false;  // analysis, no IR modification
}

bool RegisterAllocator::run(IR& ir, PassContext& ctx) {
    // No values to allocate — nothing to do.
    if (ctx.ranges.empty()) {
        ctx.assignment = Assignment{};
        return false;
    }
    // Remat loop: assignment may rewrite the IR and return nullopt,
    // requiring recomputation of live ranges.
    for (std::size_t attempt = 0, limit = ctx.ranges.size();
         attempt < limit; ++attempt) {
        auto result = assign_registers(ir, ctx.ranges, ctx);
        if (result) {
            ctx.assignment = std::move(*result);
            return false;
        }
        // Remat modified the IR — recompute live ranges and retry.
        ctx.ranges = compute_live_ranges(ir);
    }
    OPENVINO_THROW("RegisterAllocator: allocation failed after remat exhaustion");
}

bool VerifyPass::run(IR& ir, PassContext& ctx) {
    OPENVINO_ASSERT(ctx.assignment, "VerifyPass: no assignment available");
    verify(ir, ctx.ranges, *ctx.assignment, ctx);
    return false;
}

bool DumpPass::run(IR& ir, PassContext& ctx) {
    if (!ctx.dump) return false;
    std::ostringstream os;
    os << "=== " << label << ": vec_pool=" << ctx.vec_pool_indices.size()
       << " gpr_pool=" << ctx.gpr_pool_indices.size()
       << " mask_pool=" << ctx.mask_pool_indices.size()
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

PassManager build_default_pipeline() {
    PassManager pm;
    pm.add<TwoAddressPass>();
    pm.add<FoldMemoryOperandsPass>();
    pm.add<DeadDefElimPass>();
    pm.add<LiveRangeAnalysis>();
    pm.add<DumpPass>("IR before allocation");
    pm.add<RegisterAllocator>();
    pm.add<DumpPass>("IR after allocation");
    pm.add<VerifyPass>();
    pm.add<LoweringPass>();
    return pm;
}

}  // namespace ov::intel_cpu::jit_kernel_ir

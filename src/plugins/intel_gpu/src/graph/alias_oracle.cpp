// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "alias_oracle.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <set>
#include <stdexcept>
#include <unordered_map>

#include "alias_analysis.h"
#include "assign_inst.h"
#include "kv_cache_inst.h"
#include "read_value_inst.h"
#include "condition_inst.h"
#include "data_inst.h"
#include "exec_axis.h"
#include "input_layout_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "loop_inst.h"
#include "mutable_data_inst.h"
#include "primitive_inst.h"
#include "program_node.h"

namespace cldnn {

namespace {

bool is_usm(const memory& m) {
    const auto t = m.get_allocation_type();
    return t == allocation_type::usm_host || t == allocation_type::usm_shared || t == allocation_type::usm_device;
}

// [lo, hi) of a USM allocation; false for everything that is not a plain USM pointer range
bool usm_range(const memory& m, uintptr_t& lo, uintptr_t& hi) {
    if (!is_usm(m) || m.size() == 0) {
        return false;
    }
    const void* p = m.buffer_ptr();
    if (p == nullptr) {
        return false;
    }
    lo = reinterpret_cast<uintptr_t>(p);
    hi = lo + m.size();
    return true;
}

enum class share_result { yes, no, unknown };

share_result shares_allocation(const engine&, const memory& a, const memory& b) {
    if (a.size() == 0 || b.size() == 0) {
        return share_result::unknown;
    }
    // Only plain USM pointer ranges are comparable. cl_mem / images / sub-buffers are unverifiable: engine::is_the_same_buffer is not called
    // for them (it goes through the buffer accessors, which assert on locked buffers in Debug builds).
    uintptr_t alo, ahi, blo, bhi;
    if (usm_range(a, alo, ahi) && usm_range(b, blo, bhi)) {
        return (alo < bhi && blo < ahi) ? share_result::yes : share_result::no;
    }
    return share_result::unknown;
}

struct tensor_rec {
    const primitive_inst* inst = nullptr;
    const program_node* node = nullptr;
    uint16_t port = 0;
    memory::ptr mem;
    uintptr_t lo = 0, hi = 0;
    const alias_class* cls = nullptr;
    bool ctrl_flow = false;
};

}  // namespace

void network::run_alias_oracle(const exec_axis* axis_override) {
    if (!_alias_oracle_report) {
        _alias_oracle_report = std::make_unique<alias_oracle_report>();
    }
    alias_oracle_report rep;
    if (_alias_oracle_fault_for_test) {
        _alias_oracle_fault_for_test = false;
        throw std::runtime_error("alias oracle: injected fault");
    }
    const alias_analysis* aa = _program->get_alias_analysis();
    if (aa == nullptr || _internal || _program->is_body_program()) {
        rep.skipped_reason = _internal ? "internal" : _program->is_body_program() ? "body" : "no_analysis";
        *_alias_oracle_report = std::move(rep);
        return;
    }
    rep.ran = true;
    const exec_axis axis = axis_override ? *axis_override : exec_axis::build(*_program);
    rep.axis_status = axis.get_status();

    // ---- V0 ----
    if (axis.valid()) {
        for (const auto& inst : _exec_order) {
            auto it = _exec_positions.find(inst.get());
            if (it == _exec_positions.end()) {
                continue;
            }
            ++rep.v0_checked;
            const auto pos = axis.position(inst->get_node());
            if (!pos || *pos != it->second) {
                rep.hits.push_back({rule_id::V0, severity::defect, inst->id(), primitive_id(), 0, 0,
                                    "exec position " + std::to_string(it->second) + " vs axis " + (pos ? std::to_string(*pos) : std::string("none"))});
            }
        }
    }

    auto find_inst = [&](const program_node& n) -> const primitive_inst* {
        auto it = _primitives.find(n.id());
        return it == _primitives.end() ? nullptr : it->second.get();
    };
    auto out_mem = [&](const primitive_inst& inst, size_t port) -> memory::ptr {
        return port < inst.outputs_memory_count() ? inst.output_memory_ptr(port) : memory::ptr();
    };

    // ---- V3 ----
    for (const auto& e : aa->edges()) {
        if (e.act != edge_activity::ALWAYS) {
            continue;
        }
        const auto* mi = find_inst(*e.member_node);
        const auto* pi = find_inst(*e.provider_node);
        if (mi == nullptr || pi == nullptr || mi->get_flag(ExecutionFlags::SKIP)) {
            continue;
        }
        const auto mm = out_mem(*mi, e.member_port);
        const auto pm = out_mem(*pi, e.provider_port);
        if (!mm || !pm) {
            ++rep.v3_unverifiable;
            continue;
        }
        const auto r = shares_allocation(get_engine(), *mm, *pm);
        if (r == share_result::unknown) {
            ++rep.v3_unverifiable;
            continue;
        }
        ++rep.v3_checked;
        if (r == share_result::no) {
            rep.hits.push_back({rule_id::V3, severity::defect, e.member_node->id(), e.provider_node->id(), e.member_port, e.provider_port,
                                "alias edge not realized: the member does not hold the allocation of its provider"});
        }
    }

    // ---- V2 (structural) ----
    std::vector<tensor_rec> recs;
    for (const auto& kv : _primitives) {
        const primitive_inst& inst = *kv.second;
        const program_node& n = inst.get_node();
        if (n.is_type<input_layout>() || n.is_type<data>() || n.is_type<mutable_data>()) {
            continue;   // external input handles and constants
        }
        for (size_t port = 0; port < inst.outputs_memory_count(); ++port) {
            const alias_class* cls = aa->class_of(n, static_cast<int32_t>(port));
            auto mem = inst.output_memory_ptr(port);
            if (cls == nullptr || !mem || mem->size() == 0) {
                continue;
            }
            // exclusions (Appendix B3)
            const bool bound_output = inst.is_output() && (_bound_user_output_memory.count(n.id()) != 0 || _output_memory_blocks.count(n.id()) != 0 ||
                                                           _output_remote_mem_ptrs.count(n.id()) != 0);
            const bool skipped_other = inst.get_flag(ExecutionFlags::SKIP) && !(n.can_be_optimized() || n.is_runtime_skippable());
            if ((cls->domain & dm_persistent) || bound_output || skipped_other) {
                ++rep.v2_excluded;
                continue;
            }
            tensor_rec r;
            r.inst = &inst;
            r.node = &n;
            r.port = static_cast<uint16_t>(port);
            r.mem = mem;
            r.cls = cls;
            for (const auto& m : cls->members) {
                if (m.node->is_type<condition>() || m.node->is_type<loop>()) {
                    r.ctrl_flow = true;
                }
            }
            if (!usm_range(*mem, r.lo, r.hi)) {
                ++rep.v2_unverifiable;
                continue;
            }
            recs.push_back(std::move(r));
        }
    }
    std::sort(recs.begin(), recs.end(), [](const tensor_rec& a, const tensor_rec& b) {
        return a.lo != b.lo ? a.lo < b.lo : (a.node->get_unique_id() != b.node->get_unique_id() ? a.node->get_unique_id() < b.node->get_unique_id() : a.port < b.port);
    });

    // the uid that actually allocated the tensor (N5): allocators own it, zero-alloc members use the class root, MAY members use their own
    // uid unless they sit in the allocation of the root
    auto allocator_uid = [&](const tensor_rec& t) -> size_t {
        const alias_member* me = nullptr;
        for (const auto& m : t.cls->members) {
            if (m.node == t.node && m.id.index == t.port) {
                me = &m;
            }
        }
        const size_t own = t.node->get_unique_id();
        const size_t root = t.cls->root_node ? t.cls->root_node->get_unique_id() : own;
        if (me == nullptr || me->role == alias_role::allocator) {
            return own;
        }
        if (me->role == alias_role::zero_alloc_member) {
            return root;
        }
        // may_allocate_member
        if (t.cls->root_node != nullptr) {
            if (const auto* ri = find_inst(*t.cls->root_node)) {
                const auto rm = out_mem(*ri, t.cls->root.index);
                if (rm && shares_allocation(get_engine(), *t.mem, *rm) == share_result::yes) {
                    return root;
                }
            }
        }
        return own;
    };

    std::unordered_map<const memory*, std::vector<size_t>> pool_cache;
    auto pool_users = [&](const memory& m) -> const std::vector<size_t>& {
        auto it = pool_cache.find(&m);
        if (it == pool_cache.end()) {
            it = pool_cache.emplace(&m, get_memory_pool().get_user_uids(m, get_id())).first;
        }
        return it->second;
    };
    for (size_t i = 0; i < recs.size(); ++i) {
        for (size_t j = i + 1; j < recs.size() && recs[j].lo < recs[i].hi; ++j) {
            const auto& a = recs[i];
            const auto& b = recs[j];
            if (a.cls == b.cls) {
                ++rep.v2_by_class;
                continue;
            }
            ++rep.v2_pairs;
            if (a.ctrl_flow || b.ctrl_flow) {
                rep.hits.push_back({rule_id::V2, severity::info, a.node->id(), b.node->id(), a.port, b.port, "ctrl_flow output (condition / loop re-points its outputs)"});
                continue;
            }
            const size_t ua = allocator_uid(a);
            const size_t ub = allocator_uid(b);
            std::set<size_t> block;
            for (const auto* t : {&a, &b}) {   // N6: union over both memory objects; the pool is scanned once per memory object
                for (const auto uid : pool_users(*t->mem)) {
                    block.insert(uid);
                }
            }
            if (block.count(ua) && block.count(ub)) {
                ++rep.v2_by_pool;
                continue;
            }
            const bool ambiguous = aa->uid_is_ambiguous(static_cast<uint32_t>(ua)) || aa->uid_is_ambiguous(static_cast<uint32_t>(ub));
            rep.hits.push_back({rule_id::V2, ambiguous ? severity::info : severity::defect, a.node->id(), b.node->id(), a.port, b.port,
                                std::string("shared allocation of two different classes is not explained by the pool") + (ambiguous ? " (ambiguous uid)" : "")});
        }
    }

    // ---- V2b: lifetime on the actually bound memory (design 19, C6). Independent of alias_analysis, node flags and class_lifetimes. ----
    {
        struct wrec {
            uintptr_t lo = 0, hi = 0;
            const primitive_inst* inst = nullptr;
            bool padded = false;
        };
        const auto not_judged_type = [](const program_node& n) {
            return n.is_type<condition>() || n.is_type<loop>() || n.is_type<assign>() || n.is_type<read_value>() || n.is_type<kv_cache>() ||
                   n.is_type<mutable_data>();
        };
        const auto external_type = [](const program_node& n) {
            return n.is_type<input_layout>() || n.is_type<data>();
        };
        const auto position_of = [&](const primitive_inst* i) -> int64_t {
            auto it = _exec_positions.find(i);
            return it == _exec_positions.end() ? -1 : static_cast<int64_t>(it->second);
        };
        // executes a kernel / writes memory this round
        const auto writer = [&](const primitive_inst& i) {
            const auto& n = i.get_node();
            return position_of(&i) >= 0 && !i.get_flag(ExecutionFlags::SKIP) && !i.can_be_optimized() && !external_type(n) && !not_judged_type(n);
        };
        size_t max_pos = 0;
        for (const auto& kv : _exec_positions) {
            max_pos = std::max(max_pos, kv.second);
        }
        std::vector<std::vector<wrec>> writes_at(max_pos + 1);
        const auto add_write = [&](const primitive_inst& inst, size_t pos, const memory::ptr& m, bool padded) {
            wrec w;
            if (!m || !usm_range(*m, w.lo, w.hi)) {
                if (m && m->size() != 0) {
                    ++rep.v2b_unverifiable;
                }
                return;
            }
            w.inst = &inst;
            w.padded = padded;
            writes_at[pos].push_back(w);
            ++rep.v2b_writes;
        };
        for (const auto& inst : _exec_order) {
            const primitive_inst& i = *inst;
            if (not_judged_type(i.get_node())) {
                ++rep.v2b_excluded;
                continue;
            }
            if (!writer(i)) {
                continue;
            }
            const size_t pos = static_cast<size_t>(position_of(&i));
            for (size_t port = 0; port < i.outputs_memory_count(); ++port) {
                bool padded = false;
                try {
                    padded = static_cast<bool>(i.get_output_layout(port).data_padding);
                } catch (...) {
                    padded = true;
                }
                add_write(i, pos, i.output_memory_ptr(port), padded);
            }
            for (const auto& m : i.get_intermediates_memories()) {
                add_write(i, pos, m, false);
            }
        }

        const auto overlap = [](uintptr_t alo, uintptr_t ahi, uintptr_t blo, uintptr_t bhi, uintptr_t& olo, uintptr_t& ohi) {
            olo = std::max(alo, blo);
            ohi = std::min(ahi, bhi);
            return olo < ohi;
        };
        std::set<std::pair<const primitive_inst*, const primitive_inst*>> reported;
        // range R of the data that the reader at position `rpos` takes from the output of `d`
        std::function<void(uintptr_t, uintptr_t, const primitive_inst*, const primitive_inst&, int64_t, int)> judge = [&](uintptr_t rlo, uintptr_t rhi,
                                                                                                                         const primitive_inst* d,
                                                                                                                         const primitive_inst& reader,
                                                                                                                         int64_t rpos, int depth) {
            if (d == nullptr || depth > 64) {
                return;
            }
            const auto& dn = d->get_node();
            if (not_judged_type(dn)) {
                return;
            }
            int64_t dpos = -1;   // external data (input_layout / data) exists before the first position
            if (!external_type(dn)) {
                if (writer(*d)) {
                    dpos = position_of(d);
                } else {
                    // executes nothing (skipped / can_be_optimized): the data lives in the memories of its inputs
                    for (size_t j = 0; j < d->inputs_memory_count() && j < d->dependencies().size(); ++j) {
                        const auto m = d->dep_memory_ptr(j);
                        uintptr_t mlo, mhi, olo, ohi;
                        if (m && usm_range(*m, mlo, mhi) && overlap(rlo, rhi, mlo, mhi, olo, ohi)) {
                            judge(olo, ohi, d->dependencies()[j].first, reader, rpos, depth + 1);
                        }
                    }
                    return;
                }
            }
            for (int64_t q = dpos + 1; q < rpos && q <= static_cast<int64_t>(max_pos); ++q) {
                for (const auto& w : writes_at[static_cast<size_t>(q)]) {
                    uintptr_t olo, ohi;
                    if (w.inst == &reader || !overlap(rlo, rhi, w.lo, w.hi, olo, ohi)) {
                        continue;
                    }
                    if (!reported.emplace(w.inst, &reader).second) {
                        continue;
                    }
                    bool d_padded = false;
                    try {
                        for (size_t port = 0; port < d->outputs_memory_count(); ++port) {
                            d_padded = d_padded || static_cast<bool>(d->get_output_layout(port).data_padding);
                        }
                    } catch (...) {
                        d_padded = true;
                    }
                    const bool slots = w.padded && d_padded;
                    rep.v2b_hits.push_back({rule_id::V2, slots ? severity::info : severity::defect, w.inst->id(), reader.id(), 0, 0,
                                            "V2b: " + w.inst->id() + " writes [position " + std::to_string(q) + "] into memory that " + reader.id() + " reads at position " +
                                                std::to_string(rpos) + " from " + d->id() + " (produced at position " + std::to_string(dpos) + ")" +
                                                (slots ? " (padded slots of one group: not separable by range)" : "")});
                }
            }
        };
        for (const auto& inst : _exec_order) {
            const primitive_inst& x = *inst;
            if (!writer(x)) {
                continue;
            }
            const int64_t xpos = position_of(&x);
            for (size_t i = 0; i < x.inputs_memory_count() && i < x.dependencies().size(); ++i) {
                const auto m = x.dep_memory_ptr(i);
                uintptr_t lo, hi;
                if (!m || m->size() == 0) {
                    continue;
                }
                if (!usm_range(*m, lo, hi)) {
                    ++rep.v2b_unverifiable;
                    continue;
                }
                ++rep.v2b_reads_checked;
                judge(lo, hi, x.dependencies()[i].first, x, xpos, 0);
            }
        }
    }
    *_alias_oracle_report = std::move(rep);
}

}  // namespace cldnn

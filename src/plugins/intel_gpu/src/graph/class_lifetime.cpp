// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "class_lifetime.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

#include "data_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "program_node.h"
#include "selective_ssm_inst.h"
#include "shape_of_inst.h"

namespace cldnn {

namespace {

const class_interval kUnknownInterval{};
const std::vector<const program_node*> kNoNodes{};

}  // namespace

inorder_lifetime_gate_result evaluate_inorder_lifetime_gate(const program& p, const exec_axis& axis) {
    inorder_lifetime_gate_result r;
    const auto& cfg = p.get_config();
    if (!cfg.get_enable_inorder_lifetime_restrictions()) {
        return r;   // switch_off
    }
    if (cfg.get_queue_type() != QueueTypes::in_order) {
        r.reason = "queue_not_in_order";
    } else if (p.get_engine().type() != engine_types::ocl || p.get_engine().runtime_type() != runtime_types::ocl) {
        r.reason = "backend_not_ocl";
    } else if (p.is_internal_program()) {
        r.reason = "internal_program";
    } else if (p.is_body_program()) {
        r.reason = "body_program";
    } else if (!axis.valid()) {
        r.reason = "exec_axis_not_valid";
    } else {
        r.open = true;
        r.reason = "open";
    }
    return r;
}

size_t class_lifetimes::index_of(const alias_class& k) const {
    if (m_aa == nullptr) {
        return std::numeric_limits<size_t>::max();
    }
    const auto& all = m_aa->classes();
    if (all.empty() || &k < all.data() || &k >= all.data() + all.size()) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(&k - all.data());
}

class_lifetimes class_lifetimes::build(const program& p, const alias_analysis& aa, const exec_axis& axis) {
    class_lifetimes lt;
    lt.m_aa = &aa;
    lt.m_axis = axis;
    const auto& classes = aa.classes();
    lt.m_intervals.assign(classes.size(), class_interval{});
    lt.m_masks.assign(classes.size(), 0u);
    lt.m_allocators.assign(classes.size(), {});

    for (size_t ci = 0; ci < classes.size(); ++ci) {
        const auto& c = classes[ci];
        uint32_t mask = 0;
        auto& iv = lt.m_intervals[ci];

        // ---- interval ----
        if (!axis.valid()) {
            mask |= li_no_position;
        } else {
            bool known = true, missing = false, any_pos = false, inf = (c.domain & dm_user_output) != 0;
            uint32_t lo = std::numeric_limits<uint32_t>::max(), hi = 0;
            auto touch = [&](const program_node* n, bool is_reader) {
                const auto pos = axis.position(*n);
                if (!pos) {
                    // data / constant members are not executable: they carry no position and do not bound the interval
                    if (!is_reader && (n->is_type<data>() || n->is_constant())) {
                        return;
                    }
                    known = false;
                    missing = true;
                    return;
                }
                any_pos = true;
                if (!is_reader) {
                    lo = std::min(lo, *pos);
                }
                hi = std::max(hi, *pos);
                if (!is_reader && n->is_output()) {
                    inf = true;
                }
            };
            for (const auto& m : c.members) {
                touch(m.node, false);
            }
            for (const auto* rd : c.readers) {
                touch(rd, true);
            }
            if (!any_pos) {
                known = false;
                missing = true;
            }
            if (missing) {
                mask |= li_no_position;
            }
            iv.known = known;
            if (known) {
                // only non-executable members (data / constant): nothing is written on the axis, the first reader starts the interval
                if (lo == std::numeric_limits<uint32_t>::max()) {
                    lo = hi;
                }
                iv.birth = lo;
                iv.death = hi;
                iv.death_inf = inf;
            }
        }
        if (!iv.known) {
            mask |= li_unknown_interval;
        }

        // ---- eligibility ----
        if (c.has_fallback()) {
            mask |= li_fallback;
        }
        if (c.domain & dm_external_root) {
            mask |= li_external;
        }
        if (c.domain & dm_persistent) {
            mask |= li_persistent;
        }
        for (const auto* rd : c.readers) {
            if (cpu_state(*rd) != cpu_none) {
                mask |= li_cpu_reader;
            }
            if (rd->is_type<shape_of>() || rd->is_in_shape_of_subgraph()) {
                mask |= li_shape_of_reader;
            }
        }
        for (const auto& m : c.members) {
            aa.for_each_tensor_user(*m.node, m.id.index, [&](const program_node& u, size_t) {
                if (u.is_type<shape_of>()) {
                    mask |= li_shape_of_reader;
                }
            });
            if (aa.uid_is_ambiguous(m.id.uid)) {
                mask |= li_ambiguous_uid;
            }
            if ((m.role == alias_role::allocator || m.role == alias_role::may_allocate_member) && m.node->may_use_mempool()) {
                auto& v = lt.m_allocators[ci];
                if (std::find(v.begin(), v.end(), m.node) == v.end()) {
                    v.push_back(m.node);
                }
            }
        }
        lt.m_masks[ci] = mask;
    }

    // C5 (F27): selective_ssm re-points its output 1 to input 5 without an alias edge: its output classes and the whole class of input 5
    // (owner and views included) are unmodeled.
    auto mark_unmodeled = [&](const program_node& n, int32_t port) {
        const auto* c = aa.class_of(n, std::max(0, port));
        if (c != nullptr) {
            const size_t i = lt.index_of(*c);
            if (i != std::numeric_limits<size_t>::max()) {
                lt.m_masks[i] |= li_unmodeled_alias;
            }
        }
    };
    for (const auto* n : p.get_processing_order()) {
        if (!n->is_type<selective_ssm>()) {
            continue;
        }
        for (size_t port = 0; port < n->get_outputs_count(); ++port) {
            mark_unmodeled(*n, static_cast<int32_t>(port));
        }
        if (n->get_dependencies().size() > 5) {
            const auto dep = n->get_dependency_with_port(5);
            mark_unmodeled(*dep.first, dep.second);
        }
    }
    return lt;
}

const class_interval& class_lifetimes::interval_of(const alias_class& k) const {
    const size_t i = index_of(k);
    return i == std::numeric_limits<size_t>::max() ? kUnknownInterval : m_intervals[i];
}

class_interval class_lifetimes::member_interval(const program_node& n, uint16_t port) const {
    class_interval r;
    if (m_aa == nullptr) {
        return r;
    }
    const auto pos = m_axis.position(n);
    if (!pos) {
        return r;
    }
    bool known = true;
    int64_t hi = *pos;
    bool inf = false;
    std::vector<std::pair<const program_node*, uint16_t>> work{{&n, port}};
    std::set<std::pair<const program_node*, uint16_t>> seen;
    while (!work.empty()) {
        const auto t = work.back();
        work.pop_back();
        if (!seen.insert(t).second) {
            continue;
        }
        inf = inf || t.first->is_output();
        if (auto tp = m_axis.position(*t.first)) {
            hi = std::max<int64_t>(hi, *tp);
        }
        m_aa->for_each_tensor_user(*t.first, t.second, [&](const program_node& u, size_t) {
            if (u.is_type<shape_of>()) {
                return;
            }
            const auto up = m_axis.position(u);
            if (!up) {
                known = false;
            } else {
                hi = std::max<int64_t>(hi, *up);
            }
        });
        for (const size_t ei : m_aa->edges_into(*t.first, t.second)) {
            const auto& e = m_aa->edges()[ei];
            work.emplace_back(e.member_node, e.member_port);
        }
    }
    r.known = known;
    if (known) {
        r.birth = *pos;
        r.death = static_cast<uint32_t>(hi);
        r.death_inf = inf;
    }
    return r;
}

uint32_t class_lifetimes::class_ineligible_mask(const alias_class& k) const {
    const size_t i = index_of(k);
    return i == std::numeric_limits<size_t>::max() ? (li_no_class | li_unknown_interval) : m_masks[i];
}

uint32_t class_lifetimes::node_ineligible_mask(const program_node& n) const {
    if (m_aa == nullptr || !valid()) {
        return li_no_position;
    }
    uint32_t mask = 0;
    if (!m_axis.position(n)) {
        mask |= li_no_position;
    }
    if (m_aa->uid_is_ambiguous(static_cast<uint32_t>(n.get_unique_id()))) {
        mask |= li_ambiguous_uid;
    }
    const size_t ports = std::max<size_t>(1, n.get_outputs_count());
    for (size_t port = 0; port < ports; ++port) {
        const auto* c = m_aa->class_of(n, static_cast<int32_t>(port));
        if (c == nullptr) {
            mask |= li_no_class;
        } else {
            mask |= class_ineligible_mask(*c);
        }
    }
    return mask;
}

bool class_lifetimes::classes_disjoint(const program_node& a, const program_node& b) const {
    if (m_aa == nullptr) {
        return false;
    }
    const size_t pa = std::max<size_t>(1, a.get_outputs_count());
    const size_t pb = std::max<size_t>(1, b.get_outputs_count());
    for (size_t i = 0; i < pa; ++i) {
        const auto* ca = m_aa->class_of(a, static_cast<int32_t>(i));
        if (ca == nullptr) {
            continue;
        }
        for (size_t j = 0; j < pb; ++j) {
            if (m_aa->class_of(b, static_cast<int32_t>(j)) == ca) {
                return false;
            }
        }
    }
    return true;
}

const std::vector<const program_node*>& class_lifetimes::pool_allocators(const alias_class& k) const {
    const size_t i = index_of(k);
    return i == std::numeric_limits<size_t>::max() ? kNoNodes : m_allocators[i];
}

}  // namespace cldnn

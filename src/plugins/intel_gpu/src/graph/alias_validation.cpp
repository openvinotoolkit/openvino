// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "alias_validation.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>

#include "assign_inst.h"
#include "concatenation_inst.h"
#include "condition_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "kv_cache_inst.h"
#include "loop_inst.h"
#include "lora_inst.h"
#include "program_helpers.h"
#include "program_node.h"
#include "read_value_inst.h"
#include "shape_of_inst.h"
#include "stateless_kv_inst.h"

namespace cldnn {

namespace {

using tkey = std::pair<const program_node*, uint16_t>;

bool is_inplace_kind(alias_kind k) {
    return k == alias_kind::MUST_INPLACE || k == alias_kind::MAY_INPLACE;
}

bool is_must_kind(alias_kind k) {
    return k == alias_kind::MUST_VIEW || k == alias_kind::MUST_INPLACE || k == alias_kind::MUST_SUBREGION;
}

// Storage identity written by a persistent / external member (the key of persistent_binding).
storage_key storage_key_of(const program_node& n) {
    storage_key k;
    if (n.is_type<read_value>()) {
        k.k = storage_key::kind::variable;
        k.variable_id = n.as<read_value>().get_primitive()->variable_id;
    } else if (n.is_type<assign>()) {
        k.k = storage_key::kind::variable;
        k.variable_id = n.as<assign>().get_primitive()->variable_id;
    } else if (n.is_type<kv_cache>()) {
        k.k = storage_key::kind::variable;
        k.variable_id = n.as<kv_cache>().get_primitive()->variable_info.variable_id;
    } else if (n.is_type<stateless_kv>() && !n.get_dependencies().empty()) {
        k.k = storage_key::kind::external_node;
        k.node = program_helpers::resolve_buffer_owner(*n.get_dependency_with_port(0).first, program_helpers::view_resolution::settled_only).owner;
    } else {
        k.k = storage_key::kind::external_node;
        k.node = &n;
    }
    return k;
}

bool registered_writer(const alias_analysis& aa, const storage_key& key, const program_node& w) {
    for (const auto& b : aa.persistent_bindings()) {
        if (!(b.key == key)) {
            continue;
        }
        for (const auto& pw : b.writers) {
            if (pw.writer == &w) {
                return true;
            }
        }
    }
    return false;
}

// ------------------------------------------------------------------------------------------------------------------------------
// version set BFS (Appendix B2)
// ------------------------------------------------------------------------------------------------------------------------------
enum class vstate : uint8_t { EQUAL, CONTAINED, COVERING, COVERING_VIA_SLOT, OVERLAP };

struct vnode {
    tkey t;
    vstate st;
    int32_t parent;
};

}  // namespace

war_result war_safe(const alias_analysis& aa, const exec_axis& axis, const war_query& q) {
    war_result res;
    uint32_t& reasons = res.reasons;
    auto finish = [&]() {
        if (reasons & war_unsafe_mask) {
            res.verdict = war_verdict::Unsafe;
        } else if (reasons & war_unknown_mask) {
            res.verdict = war_verdict::Unknown;
        } else {
            res.verdict = war_verdict::Safe;
        }
        return res;
    };
    if (q.writer == nullptr) {
        reasons |= wr_bad_query;
        return finish();
    }
    const program_node& W = *q.writer;
    const program& prog = W.get_program();

    // the tensors overwritten by W: the dependency it reads in place, or (persistent copy writers, N1) the aliases of the variable storage
    std::vector<tkey> starts;
    const persistent_binding* copy_binding = nullptr;
    if (const auto* b = aa.binding_written_by(W)) {
        for (const auto& w : b->writers) {
            if (w.writer == &W && w.kind == persistent_writer_kind::assign_copy) {
                copy_binding = b;
            }
        }
    }
    if (copy_binding != nullptr) {
        for (const auto& a : copy_binding->aliases) {
            starts.emplace_back(a.first, a.second);
        }
    } else {
        if (q.dep_index < 0 || static_cast<size_t>(q.dep_index) >= W.get_dependencies().size()) {
            reasons |= wr_bad_query;
            return finish();
        }
        const auto dep = W.get_dependency_with_port(static_cast<size_t>(q.dep_index));
        starts.emplace_back(dep.first, static_cast<uint16_t>(std::max(0, dep.second)));
        if (aa.class_of(*dep.first, std::max(0, dep.second)) == nullptr) {
            reasons |= wr_bad_query;
            return finish();
        }
    }

    // 0. program level: the order checks are skipped (not the order independent evidence) when there is no usable axis
    bool order_ok = axis.valid();
    if (prog.get_config().get_queue_type() != QueueTypes::in_order) {
        reasons |= wr_oooq;
        order_ok = false;
    } else if (!axis.valid()) {
        reasons |= wr_no_axis;
    }
    if (prog.is_internal_program() || prog.is_body_program()) {
        reasons |= wr_body_or_internal;
        order_ok = false;
    }

    // 1. writer level
    if (cpu_state(W) != cpu_none) {
        reasons |= wr_cpu_writer;
    }
    if (W.is_in_shape_of_subgraph()) {
        reasons |= wr_shape_of_member;
    }
    if (W.is_type<stateless_kv>()) {
        reasons |= wr_runtime_guarded;   // primitive_inst.cpp keeps a runtime private copy guard
    }
    if (W.is_type<lora>() || aa.tensor_is_opaque(W, q.out_port)) {
        reasons |= wr_opaque;
    }
    const auto wpos = axis.position(W);
    res.writer_pos = wpos ? static_cast<int64_t>(*wpos) : -1;
    if (order_ok && !wpos) {
        reasons |= wr_writer_off_axis;   // no position for the writer: lateness of readers cannot be judged, never Safe
        order_ok = false;
    }

    // 3. version set. A version boundary is an in-place edge or an edge whose member is a registered in-place writer (kv cbo view edge).
    auto is_boundary = [&](const alias_edge& e) {
        if (is_inplace_kind(e.kind)) {
            return true;
        }
        if (const auto* b = aa.binding_written_by(*e.member_node)) {
            for (const auto& w : b->writers) {
                if (w.writer == e.member_node && w.kind == persistent_writer_kind::kv_cache_inplace) {
                    return true;
                }
            }
        }
        return false;
    };
    std::vector<vnode> nodes;
    std::set<std::pair<tkey, vstate>> seen;
    std::vector<size_t> bfs;
    auto push = [&](const tkey& t, vstate st, int32_t parent) {
        if (t.first == &W) {
            return;   // the output of W is the new version
        }
        if (!seen.insert({t, st}).second) {
            return;
        }
        nodes.push_back({t, st, parent});
        bfs.push_back(nodes.size() - 1);
    };
    auto on_path = [&](int32_t from, const tkey& t) {
        for (int32_t i = from; i >= 0; i = nodes[static_cast<size_t>(i)].parent) {
            if (nodes[static_cast<size_t>(i)].t == t) {
                return true;
            }
        }
        return false;
    };
    for (const auto& s : starts) {
        push(s, vstate::EQUAL, -1);
    }
    for (size_t qi = 0; qi < bfs.size(); ++qi) {
        const size_t cur = bfs[qi];
        const tkey t = nodes[cur].t;
        const vstate st = nodes[cur].st;
        // up: t is a member, go to its providers
        for (size_t ei : aa.edges_from(*t.first, t.second)) {
            const auto& e = aa.edges()[ei];
            if (is_boundary(e)) {
                continue;
            }
            vstate nst;
            const bool static_slot = e.origin == edge_origin::concat_slot && e.off == offset_tag::LAYOUT_PADDING && e.act == edge_activity::ALWAYS;
            if (st == vstate::EQUAL || st == vstate::COVERING || st == vstate::COVERING_VIA_SLOT) {
                // COVERING_VIA_SLOT (the only state that allows sibling pruning) is reached only through a static concat slot edge going up;
                // any other edge (view, crop, runtime slot) gives plain COVERING, whose relation to the siblings of a holder is unknown.
                if (static_slot) {
                    nst = vstate::COVERING_VIA_SLOT;
                } else if (st == vstate::COVERING_VIA_SLOT) {
                    nst = vstate::COVERING;
                } else {
                    nst = e.off == offset_tag::ZERO ? st : vstate::COVERING;
                }
            } else {
                nst = vstate::OVERLAP;
            }
            push({e.provider_node, e.provider_port}, nst, static_cast<int32_t>(cur));
        }
        // down: t is a provider, go to its members
        for (size_t ei : aa.edges_into(*t.first, t.second)) {
            const auto& e = aa.edges()[ei];
            if (is_boundary(e)) {
                continue;
            }
            const tkey m{e.member_node, e.member_port};
            if (on_path(static_cast<int32_t>(cur), m)) {
                continue;   // reached already, through the chain that led here
            }
            if (st == vstate::COVERING_VIA_SLOT && e.origin == edge_origin::concat_slot && e.off == offset_tag::LAYOUT_PADDING && e.act == edge_activity::ALWAYS) {
                continue;   // static sibling slots do not intersect the slot the walk came from
            }
            push(m, (st == vstate::COVERING || st == vstate::COVERING_VIA_SLOT || st == vstate::OVERLAP) ? vstate::OVERLAP : vstate::CONTAINED, static_cast<int32_t>(cur));
        }
    }
    std::set<tkey> version;
    for (const auto& n : nodes) {
        version.insert(n.t);
    }

    // 2. R4: external / persistent roots of the physical class (crosses in-place edges)
    {
        std::set<const program_node*> done;
        auto check_root = [&](const program_node& rn) {
            if (!done.insert(&rn).second || !program_helpers::is_external_buffer_root(rn)) {
                return;
            }
            if (registered_writer(aa, storage_key_of(rn), W)) {
                return;
            }
            reasons |= is_persistent_type(rn) ? wr_persistent_unregistered : wr_external_root;
            bool in_version = false;
            for (const auto& vt : version) {
                if (vt.first == &rn) {
                    in_version = true;
                    break;
                }
            }
            if (!in_version) {
                reasons |= wr_via_upstream_inplace;
            }
        };
        for (const auto& s : starts) {
            const alias_class* k = aa.class_of(*s.first, s.second);
            if (k == nullptr) {
                continue;
            }
            for (const auto& m : k->members) {
                if (m.role == alias_role::allocator || program_helpers::is_external_buffer_root(*m.node)) {
                    check_root(*m.node);
                }
            }
        }
    }

    // member checks
    for (const auto& vt : version) {
        const program_node& n = *vt.first;
        if (aa.tensor_is_opaque(n, vt.second)) {
            reasons |= wr_opaque;
        }
        if (cpu_state(n) != cpu_none) {
            reasons |= wr_cpu_member;
        }
        if (n.is_in_shape_of_subgraph()) {
            reasons |= wr_shape_of_member;
        }
        if (program_helpers::is_or_becomes_output(n)) {
            reasons |= wr_visible_version;   // R5a: virtual last reader
        }
    }

    // 4. readers
    bool any_reader = false;
    auto witness = [&](const program_node& u, std::optional<uint32_t> pu, const tkey& via) {
        if (res.witnesses.size() < 4) {
            res.witnesses.push_back({&u, pu ? static_cast<int64_t>(*pu) : -1, via.first, via.second});
        }
    };
    for (const auto& vt : version) {
        aa.for_each_tensor_user(*vt.first, vt.second, [&](const program_node& u, size_t d) {
            if (u.is_type<shape_of>()) {
                return;
            }
            if (&u == &W) {
                const bool is_the_inplace_read = copy_binding != nullptr || (starts.size() == 1 && vt.first == starts[0].first &&
                                                                              vt.second == starts[0].second && static_cast<int32_t>(d) == q.dep_index);
                if (is_the_inplace_read) {
                    return;
                }
                reasons |= wr_self_read_other_port;   // F11
                witness(u, wpos, vt);
                return;
            }
            any_reader = true;
            if (cpu_state(u) != cpu_none) {
                reasons |= wr_cpu_reader;
                return;
            }
            if (u.is_in_shape_of_subgraph()) {
                reasons |= wr_shape_of_subgraph_reader;
                return;
            }
            if (u.is_type<loop>() || u.is_type<condition>()) {
                reasons |= wr_ctrl_flow_reader;
                return;
            }
            if (!order_ok) {
                return;
            }
            const auto pu = axis.position(u);
            if (!pu) {
                reasons |= wr_reader_off_axis;
                return;
            }
            if (wpos && *pu > *wpos) {
                reasons |= wr_late_reader;
                witness(u, pu, vt);
            }
        });
    }
    if (!any_reader && !(reasons & (war_unsafe_mask | war_unknown_mask))) {
        reasons |= wr_no_other_reader;
    }
    return finish();
}

std::vector<war_query> enumerate_inplace_writers(const alias_analysis& aa) {
    std::vector<war_query> out;
    std::set<std::pair<const program_node*, int32_t>> seen;
    for (const auto& e : aa.edges()) {
        if (!is_inplace_kind(e.kind) || e.dep_index < 0) {
            continue;
        }
        if (seen.insert({e.member_node, e.dep_index}).second) {
            out.push_back(war_query{e.member_node, e.dep_index, e.member_port});
        }
    }
    for (const auto& b : aa.persistent_bindings()) {
        for (const auto& w : b.writers) {
            // assign (copy writer) and kv_cache (always counted as in place, also when it is a cbo node with only a view edge)
            const bool counted = w.kind == persistent_writer_kind::assign_copy || w.kind == persistent_writer_kind::kv_cache_inplace;
            if (counted && seen.insert({w.writer, 0}).second) {
                out.push_back(war_query{w.writer, 0, w.port});
            }
        }
    }
    return out;
}

const char* to_string(rule_id id) {
    switch (id) {
    case rule_id::R1:
        return "R1";
    case rule_id::R2:
        return "R2";
    case rule_id::R5b:
        return "R5b";
    case rule_id::R5c:
        return "R5c";
    case rule_id::V0:
        return "V0";
    case rule_id::V1:
        return "V1";
    case rule_id::V2:
        return "V2";
    case rule_id::V3:
        return "V3";
    }
    return "?";
}

const char* to_string(war_verdict v) {
    switch (v) {
    case war_verdict::Safe:
        return "Safe";
    case war_verdict::Unsafe:
        return "Unsafe";
    case war_verdict::Unknown:
        return "Unknown";
    }
    return "?";
}

size_t alias_validation_report::defect_count() const {
    size_t n = 0;
    for (const auto& kv : counts) {
        if (kv.first.second == severity::defect) {
            n += kv.second;
        }
    }
    return n;
}

size_t alias_validation_report::war_count(war_verdict v) const {
    size_t n = 0;
    for (const auto& w : war) {
        n += w.r.verdict == v ? 1 : 0;
    }
    return n;
}

std::string alias_validation_report::summary() const {
    std::ostringstream os;
    os << "[alias_validation] ran=" << ran;
    if (!skipped_reason.empty()) {
        os << " skipped=" << skipped_reason;
    }
    os << " axis=" << static_cast<int>(axis.get_status()) << " writers=" << war.size() << " (safe=" << war_count(war_verdict::Safe)
       << " unsafe=" << war_count(war_verdict::Unsafe) << " unknown=" << war_count(war_verdict::Unknown) << ")";
    for (const auto& kv : counts) {
        os << " " << to_string(kv.first.first) << (kv.first.second == severity::defect ? "/D=" : kv.first.second == severity::war ? "/W=" : "/I=")
           << kv.second;
    }
    return os.str();
}

// ------------------------------------------------------------------------------------------------------------------------------
// compile-time rules
// ------------------------------------------------------------------------------------------------------------------------------
namespace {

struct role_flags {
    bool view = false, slot = false, holder = false, inplace_w = false, inplace_t = false, out = false, ext = false, persist = false, soc = false,
         soc_shape_of = false;
    bool view_static = true, slot_static = true;
};

size_t tensor_bytes(const program_node& n, size_t port, bool& known) {
    if (!n.is_all_valid_output_layouts()) {
        known = false;
        return 0;
    }
    const auto& layouts = n.get_output_layouts();
    if (port >= layouts.size() || layouts[port].is_dynamic() || layouts[port].data_padding.is_dynamic() || layouts[port].format == format::any) {
        known = false;
        return 0;
    }
    return layouts[port].bytes_count();
}

}  // namespace

alias_validation_report validate_alias_rules(const program& p, const alias_analysis& aa) {
    alias_validation_report rep;
    if (p.is_internal_program()) {
        rep.skipped_reason = "internal";
        return rep;
    }
    if (p.is_body_program()) {
        rep.skipped_reason = "body";
        return rep;
    }
    rep.ran = true;
    rep.axis = exec_axis::build(p);

    auto add_hit = [&](rule_id id, severity sev, const program_node* a, const program_node* b, uint16_t port, std::string cell) {
        rep.hits.push_back(rule_hit{id, sev, a, b, port, std::move(cell)});
        ++rep.counts[{id, sev}];
    };

    // ---- war report of every in-place writer ----
    for (const auto& q : enumerate_inplace_writers(aa)) {
        alias_validation_report::war_entry we;
        we.q = q;
        alias_kind kind = alias_kind::NONE;
        edge_origin origin = edge_origin::view;
        for (size_t ei : aa.edges_from(*q.writer, q.out_port)) {
            const auto& e = aa.edges()[ei];
            if (is_inplace_kind(e.kind) && e.dep_index == q.dep_index) {
                kind = e.kind;
                origin = e.origin;
                break;
            }
        }
        we.kind = kind;
        we.origin = origin;
        we.r = war_safe(aa, rep.axis, q);
        rep.war.push_back(std::move(we));
    }

    // ---- R1 role matrix, R2, R5b, R5c, V1 ----
    for (const auto& cls : aa.classes()) {
        // per class statistics (R2)
        if (cls.role_conflict_causes != 0) {
            std::string cell;
            const struct {
                uint32_t bit;
                const char* name;
            } names[] = {{rc_allocators_not_one, "allocators_not_one"}, {rc_opaque, "opaque"}, {rc_member_exceeds_root, "member_exceeds_root"},
                         {rc_view_slot_pred, "view_slot_pred"}};
            for (const auto& nm : names) {
                if (cls.role_conflict_causes & nm.bit) {
                    cell += (cell.empty() ? "" : "|");
                    cell += nm.name;
                }
            }
            add_hit(rule_id::R2, severity::info, cls.root_node, nullptr, cls.root.index, cell);
        }

        bool all_must = true;
        for (const auto& m : cls.members) {
            if (m.edge_count > 1) {
                add_hit(rule_id::R2, severity::info, m.node, nullptr, m.id.index, "multi_edge");
            }
            if (!m.node->is_output() && program_helpers::is_or_becomes_output(*m.node)) {
                add_hit(rule_id::R5c, severity::info, m.node, nullptr, m.id.index, "mutable_data_propagated_output");
            }
            const program_node& n = *m.node;
            const uint16_t port = m.id.index;
            role_flags f;
            const alias_edge* first_slot = nullptr;
            for (size_t ei : aa.edges_from(n, port)) {
                const auto& e = aa.edges()[ei];
                all_must = all_must && is_must_kind(e.kind);
                if (e.origin == edge_origin::view) {
                    f.view = true;
                    f.view_static = f.view_static && is_must_kind(e.kind);
                }
                if (e.origin == edge_origin::concat_slot) {
                    f.slot = true;
                    f.slot_static = f.slot_static && is_must_kind(e.kind) && e.act == edge_activity::ALWAYS;
                    first_slot = first_slot ? first_slot : &e;
                }
                if (is_inplace_kind(e.kind)) {
                    f.inplace_w = true;
                }
            }
            for (size_t ei : aa.edges_into(n, port)) {
                const auto& e = aa.edges()[ei];
                if (e.origin == edge_origin::concat_slot) {
                    f.holder = true;
                }
                if (is_inplace_kind(e.kind)) {
                    f.inplace_t = true;
                }
            }
            f.out = program_helpers::is_or_becomes_output(n);
            f.ext = program_helpers::is_external_buffer_root(n) && !is_persistent_type(n);
            f.persist = is_persistent_type(n);
            f.soc_shape_of = n.is_in_shape_of_subgraph();
            f.soc = f.soc_shape_of || cpu_state(n) != cpu_none;
            (void)f.inplace_t;

            auto D = [&](const char* cell) { add_hit(rule_id::R1, severity::defect, &n, nullptr, port, cell); };
            auto I = [&](const char* cell) { add_hit(rule_id::R1, severity::info, &n, nullptr, port, cell); };

            // VIEW x *
            if (f.view && f.slot) {
                (f.view_static && f.slot_static) ? D("VIEW x SLOT") : I("VIEW x SLOT (MAY)");
            }
            if (f.view && f.holder) {
                D("VIEW x HOLDER");
            }
            if (f.view && f.inplace_w) {
                D("VIEW x INPLACE_W");
            }
            if (f.view && f.ext) {
                I("VIEW x EXT");
            }
            if (f.view && f.persist) {
                n.is_type<kv_cache>() ? I("VIEW x PERSIST (kv_cache)") : D("VIEW x PERSIST");
            }
            if (f.view && f.out) {
                I("VIEW x OUT");
            }
            if (f.view && f.soc) {
                I("VIEW x SOC");
            }
            // SLOT x *
            if (f.slot && f.holder) {
                bool legal = f.slot_static && first_slot != nullptr && n.is_type<concatenation>() && first_slot->provider_node->is_type<concatenation>() &&
                             n.as<concatenation>().get_primitive()->axis == first_slot->provider_node->as<concatenation>().get_primitive()->axis;
                if (!legal) {
                    D("SLOT x HOLDER");
                }
            }
            if (f.slot && f.inplace_w) {
                D("SLOT x INPLACE_W");
            }
            if (f.slot && f.out) {
                D("SLOT x OUT");
            }
            if (f.slot && f.ext) {
                D("SLOT x EXT");
            }
            if (f.slot && f.persist) {
                D("SLOT x PERSIST");
            }
            if (f.slot && f.soc) {
                f.soc_shape_of ? D("SLOT x SOC (shape_of)") : I("SLOT x SOC (cpu)");
            }
            // HOLDER x *
            if (f.holder && f.inplace_w) {
                D("HOLDER x INPLACE_W");
            }
            if (f.holder && f.out) {
                D("HOLDER x OUT");
            }
            if (f.holder && (f.ext || f.persist)) {
                D("HOLDER x EXT/PERSIST");
            }
            if (f.holder && f.soc) {
                f.soc_shape_of ? D("HOLDER x SOC (shape_of)") : I("HOLDER x SOC (cpu)");
            }
            // INPLACE_W x *
            if (f.inplace_w) {
                bool must_inplace = false;
                bool sum_origin = false;
                for (size_t ei : aa.edges_from(n, port)) {
                    const auto& e = aa.edges()[ei];
                    must_inplace = must_inplace || e.kind == alias_kind::MUST_INPLACE;
                    sum_origin = sum_origin || e.origin == edge_origin::onednn_sum;
                }
                if (f.persist && aa.binding_written_by(n) == nullptr) {
                    D("INPLACE_W x PERSIST (unregistered)");
                }
                if (f.soc) {
                    sum_origin ? D("INPLACE_W x SOC (sum)") : I("INPLACE_W x SOC");
                }
                // R5b: INPLACE_W x OUT (static MUST) and the forward view chain of the written tensor
                if (must_inplace) {
                    bool visible = f.out;
                    std::string cell = "INPLACE_W x OUT";
                    if (!visible) {
                        std::set<tkey> seen_fw;
                        std::vector<tkey> stack{{&n, port}};
                        while (!stack.empty() && !visible) {
                            const tkey cur = stack.back();
                            stack.pop_back();
                            for (size_t ei : aa.edges_into(*cur.first, cur.second)) {
                                const auto& e = aa.edges()[ei];
                                if (e.origin != edge_origin::view) {
                                    continue;
                                }
                                const tkey mt{e.member_node, e.member_port};
                                if (!seen_fw.insert(mt).second) {
                                    continue;
                                }
                                if (program_helpers::is_or_becomes_output(*e.member_node)) {
                                    visible = true;
                                    cell = "INPLACE_W forward view chain x OUT";
                                    break;
                                }
                                stack.push_back(mt);
                            }
                        }
                    }
                    if (visible) {
                        add_hit(rule_id::R5b, severity::defect, &n, nullptr, port, cell);
                    }
                } else if (f.out) {
                    I("INPLACE_W x OUT (MAY)");
                }
            }
            // OUT x EXT / PERSIST
            if (f.out && f.ext) {
                I("OUT x EXT");
            }
            if (f.out && f.persist) {
                I("OUT x PERSIST");
            }
        }

        // V1: member bytes must fit the root allocation. Defect only when every edge of the class is a MUST edge.
        bool root_known = true;
        const size_t root_bytes = cls.root_node ? tensor_bytes(*cls.root_node, cls.root.index, root_known) : 0;
        if (root_known && cls.root_node != nullptr) {
            bool all_known = true;
            const program_node* biggest = nullptr;
            size_t biggest_bytes = 0;
            for (const auto& m : cls.members) {
                bool known = true;
                const size_t bytes = tensor_bytes(*m.node, m.id.index, known);
                all_known = all_known && known;
                if (known && bytes > biggest_bytes) {
                    biggest_bytes = bytes;
                    biggest = m.node;
                }
            }
            if (all_known && biggest_bytes > root_bytes) {
                add_hit(rule_id::V1, (all_must && !is_persistent_type(*biggest)) ? severity::defect : severity::info, biggest, cls.root_node, 0, "member bytes > root bytes");
            }
        }
    }
    return rep;
}

}  // namespace cldnn

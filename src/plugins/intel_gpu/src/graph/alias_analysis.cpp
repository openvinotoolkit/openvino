// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "alias_analysis.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <unordered_set>

#include "assign_inst.h"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "input_layout_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "kv_cache_inst.h"
#include "lora_inst.h"
#include "primitive_type.h"
#include "program_helpers.h"
#include "program_node.h"
#include "read_value_inst.h"
#include "shape_of_inst.h"
#include "stateless_kv_inst.h"

namespace cldnn {

namespace {

alias_kind to_may(alias_kind k) {
    switch (k) {
    case alias_kind::MUST_VIEW:
        return alias_kind::MAY_VIEW;
    case alias_kind::MUST_INPLACE:
        return alias_kind::MAY_INPLACE;
    case alias_kind::MUST_SUBREGION:
        return alias_kind::MAY_SUBREGION;
    default:
        return k;
    }
}

bool is_may(alias_kind k) {
    return k == alias_kind::MAY_VIEW || k == alias_kind::MAY_INPLACE || k == alias_kind::MAY_SUBREGION;
}

bool is_persistent_type(const program_node& n) {
    return n.is_type<read_value>() || n.is_type<assign>() || n.is_type<kv_cache>() || n.is_type<stateless_kv>();
}

uint32_t to_uid32(const program_node& n) {
    const size_t uid = n.get_unique_id();
    OPENVINO_ASSERT(uid <= std::numeric_limits<uint32_t>::max(), "[GPU] alias_analysis: unique id of node ", n.id(), " does not fit uint32");
    return static_cast<uint32_t>(uid);
}

uint16_t checked_port(int64_t port) {
    OPENVINO_ASSERT(port >= 0 && port <= std::numeric_limits<uint16_t>::max(), "[GPU] alias_analysis: output port ", port, " out of range");
    return static_cast<uint16_t>(port);
}

constexpr int8_t cpu_none = 0, cpu_sure = 1, cpu_maybe = 2;

// Same rule the allocator uses (primitive_inst.cpp: selected_impl ? is_cpu() : preferred == cpu); dynamic nodes whose supported
// implementation set contains a CPU one are "maybe CPU".
int8_t cpu_state(const program_node& n) {
    if (n.is_dynamic()) {
        if (n.is_type<input_layout>() || n.is_type<data>()) {
            return cpu_none;
        }
        // Request only the impl type (any shape type): a primitive with a static-only CPU impl (e.g. non_max_suppression) can still land on
        // CPU for a dynamic node, because ImplementationsFactory picks a static impl from the supported set per concrete shape. Requesting
        // only cpu also keeps validate() away from the oneDNN lazy init (get_kernel_impl_params).
        return n.type()->has_impl_for(n, impl_types::cpu) ? cpu_maybe : cpu_none;
    }
    const bool is_cpu = n.get_selected_impl() ? n.get_selected_impl()->is_cpu() : n.get_preferred_impl_type() == impl_types::cpu;
    return is_cpu ? cpu_sure : cpu_none;
}

struct dsu {
    std::vector<size_t> parent;
    explicit dsu(size_t n) : parent(n) {
        std::iota(parent.begin(), parent.end(), size_t{0});
    }
    size_t find(size_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }
    void unite(size_t a, size_t b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[std::max(a, b)] = std::min(a, b);
        }
    }
};

}  // namespace

edge_activity compile_time_envelope::active(const alias_member& m) const {
    const program_node& n = *m.node;
    switch (m.kind) {
    case alias_kind::MUST_SUBREGION: {
        const auto* holder = program_helpers::in_place_concat_holder_of(n);
        return (holder && (holder->is_dynamic() || holder->is_runtime_skippable())) ? edge_activity::MAYBE : edge_activity::ALWAYS;
    }
    case alias_kind::MUST_VIEW:
        return (n.is_dynamic() || n.is_runtime_skippable()) ? edge_activity::MAYBE : edge_activity::ALWAYS;
    case alias_kind::MUST_INPLACE:
        return n.is_dynamic() ? edge_activity::MAYBE : edge_activity::ALWAYS;
    default:
        return edge_activity::ALWAYS;
    }
}

size_t alias_analysis::debug_po_rank(const program_node& n) const {
    auto it = m_po_rank.find(&n);
    return it == m_po_rank.end() ? std::numeric_limits<size_t>::max() : it->second;
}

const alias_class* alias_analysis::class_of(const program_node& n, int32_t port) const {
    OPENVINO_ASSERT(port >= 0, "[GPU] alias_analysis::class_of: negative port ", port);
    auto it = m_class_of_node.find(&n);
    if (it == m_class_of_node.end() || static_cast<size_t>(port) >= it->second.size() ||
        it->second[static_cast<size_t>(port)] == std::numeric_limits<size_t>::max()) {
        return nullptr;
    }
    return &m_classes[it->second[static_cast<size_t>(port)]];
}

const alias_class* alias_analysis::class_of(const tensor_id& id) const {
    if (m_colliding_keys.count(key_of(id)) != 0) {
        return nullptr;  // ambiguous uid projection: callers must use node identity
    }
    auto it = m_class_of_tensor.find(key_of(id));
    return it == m_class_of_tensor.end() ? nullptr : &m_classes[it->second];
}

bool alias_analysis::uid_is_ambiguous(uint32_t uid) const {
    for (const auto k : m_colliding_keys) {
        if (static_cast<uint32_t>(k >> 32) == uid) {
            return true;
        }
    }
    return false;
}

size_t alias_analysis::ambiguous_uid_count() const {
    std::unordered_set<uint32_t> uids;
    for (const auto k : m_colliding_keys) {
        uids.insert(static_cast<uint32_t>(k >> 32));
    }
    return uids.size();
}

tensor_id alias_analysis::root_of(const tensor_id& id) const {
    const auto* c = class_of(id);
    return c ? c->root : id;
}

alias_analysis alias_analysis::build(const program& p, const decision_source& src) {
    alias_analysis res;

    // 1. Snapshot of the processing order. Only used to order members and classes deterministically (not a time axis).
    std::vector<const program_node*> nodes;
    for (const auto* n : p.get_processing_order()) {
        nodes.push_back(n);
    }
    std::unordered_map<const program_node*, size_t> idx_of;
    idx_of.reserve(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        idx_of.emplace(nodes[i], i);
        res.m_po_rank.emplace(nodes[i], i);
    }

    // Output ports per node: declared count, raised by whatever port a user reads.
    std::vector<size_t> nports(nodes.size(), 1);
    for (size_t i = 0; i < nodes.size(); ++i) {
        nports[i] = std::max<size_t>(nports[i], nodes[i]->get_outputs_count());
    }
    for (const auto* n : nodes) {
        for (const auto& d : n->get_dependencies()) {
            auto it = idx_of.find(d.first);
            if (it != idx_of.end()) {
                nports[it->second] = std::max<size_t>(nports[it->second], static_cast<size_t>(std::max(0, d.second)) + 1);
            }
        }
    }
    std::vector<size_t> base(nodes.size() + 1, 0);
    for (size_t i = 0; i < nodes.size(); ++i) {
        base[i + 1] = base[i] + nports[i];
    }
    const size_t tensor_count = base.back();
    auto tensor_of = [&](size_t node_idx, int32_t port) {
        return base[node_idx] + static_cast<size_t>(std::max(0, port));
    };

    struct tensor_info {
        alias_kind kind = alias_kind::NONE;
        offset_tag off = offset_tag::ZERO;
        uint32_t providers = 0;
        bool opaque = false;
        bool slot_pred = false;   // has a slot (in-place concat) provider
    };
    std::vector<tensor_info> info(tensor_count);
    dsu sets(tensor_count);

    auto add_edge = [&](size_t member_node, int32_t member_port, alias_kind must_kind, offset_tag off, size_t provider_tensor) {
        const program_node& n = *nodes[member_node];
        alias_member probe;
        probe.id = tensor_id{to_uid32(n), tensor_space::OUTPUT, checked_port(member_port)};
        probe.kind = must_kind;
        probe.node = &n;
        const edge_activity act = src.active(probe);
        if (act == edge_activity::NEVER) {
            return;
        }
        alias_kind kind = must_kind;
        if (act == edge_activity::MAYBE) {
            kind = to_may(must_kind);
            if (off != offset_tag::ZERO) {
                off = offset_tag::RUNTIME;
            }
        }
        const size_t t = tensor_of(member_node, member_port);
        sets.unite(t, provider_tensor);
        if (info[t].providers++ == 0) {
            info[t].kind = kind;
            info[t].off = off;
        }
    };

    // 2. Alias edges.
    for (size_t i = 0; i < nodes.size(); ++i) {
        const program_node& n = *nodes[i];

        // slot of an in-place concat: the consumer is the holder (allocator), this node is a region of it.
        for (const auto* u : n.get_users()) {
            if (!(u->is_type<concatenation>() && u->can_be_optimized())) {
                continue;
            }
            auto h = idx_of.find(u);
            if (h == idx_of.end()) {
                continue;
            }
            for (const auto& d : u->get_dependencies()) {
                if (d.first != &n) {
                    continue;
                }
                add_edge(i, d.second, alias_kind::MUST_SUBREGION, offset_tag::LAYOUT_PADDING, tensor_of(h->second, 0));
                info[tensor_of(i, d.second)].slot_pred = true;
            }
        }

        // view of input 0 (settled flags only)
        if (program_helpers::is_buffer_view_of_input0(n, program_helpers::view_resolution::settled_only)) {
            const auto dep = n.get_dependency_with_port(0);
            auto d = idx_of.find(dep.first);
            if (d != idx_of.end()) {
                const offset_tag off = (n.is_type<crop>() && n.can_be_optimized()) ? offset_tag::STATIC_BYTES : offset_tag::ZERO;
                add_edge(i, 0, alias_kind::MUST_VIEW, off, tensor_of(d->second, dep.second));
            }
        }

        // oneDNN in-place writer: output lives in the buffer of the reused eltwise input. A sum is final (F5) so it is a MUST edge
        // for static nodes; a mul may still be vetoed by the impl binding (F17 G-impl), hence MAY.
        if (const int32_t reused = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(n); reused != -1) {
            bool is_sum = false;
            for (const auto& fused_op : n.get_fused_primitives()) {
                if (fused_op.is_type<eltwise>() && fused_op.deps.size() == 1 && fused_op.has_outer_dep() &&
                    fused_op.outer_dep_start_idx == reused && fused_op.typed_desc<eltwise>()->mode == eltwise_mode::sum) {
                    is_sum = true;
                    break;
                }
            }
            const auto dep = n.get_dependency_with_port(static_cast<size_t>(reused));
            auto d = idx_of.find(dep.first);
            if (d != idx_of.end()) {
                add_edge(i, 0, is_sum ? alias_kind::MUST_INPLACE : alias_kind::MAY_INPLACE, offset_tag::ZERO, tensor_of(d->second, dep.second));
            }
        }

        // stateful in-place aliases. Decision (user, 2026-11-20; errata of design 08 B.2 step 3, which only listed a cbo kv_cache):
        // EVERY kv_cache gets a MAY_INPLACE edge, because primitive_inst::do_runtime_in_place_kv_cache() (primitive_inst.cpp:2045-2112,
        // called at :2741) runs for every kv_cache and does not look at can_be_optimized.
        // The runtime target is the VARIABLE memory, not "dependency 0" as such: realloc_if_needed sets _outputs[0] = variable.get_memory()
        // on the cbo path (primitive_inst.cpp:~1239) and on the non-cbo, !is_reuse_variable_mem path (~1268-1271), and the in-place update
        // writes variable.set_layout (~2090-2092). Dependency 0 is that memory only when it resolves to the read_value of the SAME
        // variable_id (directly or through settled views); otherwise no edge is built and the node's class is a role conflict (opaque).
        // stateless_kv has no variable: its in-place target is `past`, dependency 0 (primitive_inst.cpp:~871-889).
        // A cbo kv_cache already has a settled-view edge to dependency 0; the in-place edge is added only if the two tensors are not in the
        // same class yet (union-find test), so an earlier provider (e.g. an in-place concat slot) never hides the kv alias.
        // The classes are persistent (dm_persistent, is_persistent_type()). NOTE for Phase 0.5b / 1a: in practice a kv class also contains the
        // read_value, which only has CPU implementations, so it carries fb_cpu_impl / fb_maybe_cpu and is LEGACY; STATE would only be reached
        // if that rule were lifted. war_safe must treat persistent kv classes (STATE or CPU-fallback) as an external root that only
        // registered persistent writers may write, or return Unknown.
        if (!n.get_dependencies().empty() && (n.is_type<stateless_kv>() || n.is_type<kv_cache>())) {
            const auto dep = n.get_dependency_with_port(0);
            auto d = idx_of.find(dep.first);
            bool target_ok = true;
            if (n.is_type<kv_cache>()) {
                const auto* owner = program_helpers::resolve_buffer_owner(*dep.first, program_helpers::view_resolution::settled_only).owner;
                target_ok = owner->is_type<read_value>() && owner->as<read_value>().get_primitive()->variable_id ==
                                                                n.as<kv_cache>().get_primitive()->variable_info.variable_id;
            }
            if (!target_ok) {
                for (size_t k = 0; k < nports[i]; ++k) {
                    info[tensor_of(i, static_cast<int32_t>(k))].opaque = true;  // alias target not statically known: conservative fallback
                }
            } else if (d != idx_of.end()) {
                const size_t target = tensor_of(d->second, dep.second);
                if (sets.find(tensor_of(i, 0)) != sets.find(target)) {
                    add_edge(i, 0, alias_kind::MAY_INPLACE, offset_tag::ZERO, target);
                }
            }
        }
        // Multi-output state aliases that are not statically resolvable (Phase 0.5b gap, conservative fallback instead of a guess):
        //  * compressed kv_cache: _outputs[2] / _outputs[3] are the compression scale / zero-point STATE memories of the variable
        //    (primitive_inst.cpp:~1246/1249, ~1274/1277, ~1689-1695); which dependency (if any) is the matching read_value depends on
        //    the compression configuration (get_compression_scales_inputs_num / zp inputs num), which this analysis does not model;
        //  * stateless_kv with a second output: _outputs[1] is reinterpret_buffer(kv_present) (primitive_inst.cpp:~1088), where kv_present is
        //    _outputs[0] or a node-private buffer depending on the rolling/private-present decision taken at runtime.
        // All ports of such a node get fb_role_conflict (class not promotable) until 0.5b models these edges.
        if ((n.is_type<kv_cache>() && n.as<kv_cache>().get_primitive()->compressed) || (n.is_type<stateless_kv>() && nports[i] > 1)) {
            for (size_t k = 0; k < nports[i]; ++k) {
                info[tensor_of(i, static_cast<int32_t>(k))].opaque = true;
            }
        }
        if (n.is_type<lora>() && !n.get_dependencies().empty()) {
            // LoRA may reuse the memory of its input (pass_manager.h memory_dependency_pass): opaque, class is not promotable.
            const auto dep = n.get_dependency_with_port(0);
            auto d = idx_of.find(dep.first);
            if (d != idx_of.end()) {
                add_edge(i, 0, alias_kind::MAY_INPLACE, offset_tag::ZERO, tensor_of(d->second, dep.second));
            }
            info[tensor_of(i, 0)].opaque = true;
        }
    }

    // 3. Classes: group by DSU representative, order by the PO rank of the first member.
    std::vector<size_t> rep_to_class(tensor_count, std::numeric_limits<size_t>::max());
    std::vector<std::vector<size_t>> class_tensors;
    std::vector<size_t> tensor_node(tensor_count);
    for (size_t i = 0; i < nodes.size(); ++i) {
        for (size_t k = 0; k < nports[i]; ++k) {
            tensor_node[base[i] + k] = i;
        }
    }
    for (size_t t = 0; t < tensor_count; ++t) {  // tensors are laid out in PO order, so classes come out ordered by first member
        const size_t rep = sets.find(t);
        if (rep_to_class[rep] == std::numeric_limits<size_t>::max()) {
            rep_to_class[rep] = class_tensors.size();
            class_tensors.emplace_back();
        }
        class_tensors[rep_to_class[rep]].push_back(t);
    }

    // program-level reasons
    uint32_t program_fallback = 0;
    if (p.get_config().get_queue_type() != QueueTypes::in_order) {
        program_fallback |= fb_oooq;
    }
    if (p.is_internal_program()) {
        program_fallback |= fb_internal_net;
    }
    if (p.is_body_program()) {
        program_fallback |= fb_body_ctrl_flow;
    }

    std::vector<int8_t> cpu_cache(nodes.size(), -1);
    auto node_cpu = [&](size_t node_idx) {
        if (cpu_cache[node_idx] < 0) {
            cpu_cache[node_idx] = cpu_state(*nodes[node_idx]);
        }
        return cpu_cache[node_idx];
    };
    auto cpu_bits = [](int8_t s) -> uint32_t {
        return s == cpu_sure ? fb_cpu_impl : s == cpu_maybe ? fb_maybe_cpu : 0u;
    };
    auto tensor_bytes = [&](size_t t, bool& known) -> size_t {
        const program_node& n = *nodes[tensor_node[t]];
        const size_t port = t - base[tensor_node[t]];
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
    };

    res.m_classes.reserve(class_tensors.size());
    for (const auto& tensors : class_tensors) {
        alias_class cls;
        cls.fallback = program_fallback;

        // members sorted by (PO rank, port): tensors are already in that order
        std::unordered_set<size_t> member_nodes;
        std::vector<size_t> allocators;
        for (size_t t : tensors) {
            const size_t ni = tensor_node[t];
            member_nodes.insert(ni);
            if (info[t].providers == 0) {
                allocators.push_back(t);
            }
        }
        size_t root_t = tensors.front();
        if (allocators.size() == 1) {
            root_t = allocators.front();
        } else {
            cls.fallback |= fb_role_conflict;
            if (!allocators.empty()) {
                root_t = allocators.front();
            }
        }

        bool sizes_known = true;
        const size_t root_bytes = tensor_bytes(root_t, sizes_known);
        size_t max_bytes = 0;

        for (size_t t : tensors) {
            const size_t ni = tensor_node[t];
            const program_node& n = *nodes[ni];
            alias_member m;
            m.id = tensor_id{to_uid32(n), tensor_space::OUTPUT, checked_port(static_cast<int64_t>(t - base[ni]))};
            m.node = &n;
            if (info[t].providers == 0) {
                m.kind = alias_kind::NONE;
                m.role = alias_role::allocator;
            } else {
                m.kind = info[t].kind;
                m.off = info[t].off;
                m.role = is_may(m.kind) ? alias_role::may_allocate_member : alias_role::zero_alloc_member;
            }
            cls.members.push_back(m);
            if (t == root_t) {
                cls.root = m.id;
                cls.root_node = &n;
            }

            cls.fallback |= cpu_bits(node_cpu(ni));
            if (n.is_in_shape_of_subgraph()) {
                cls.fallback |= fb_shape_of;
            }
            if (info[t].opaque) {
                cls.fallback |= fb_role_conflict;
            }
            // a concat predecessor that is itself a view / skippable node has two providers (cascaded concat is allowed)
            if (info[t].slot_pred && (n.can_be_optimized() || n.is_runtime_skippable()) && !n.is_type<concatenation>()) {
                cls.fallback |= fb_role_conflict;
            }
            if (is_persistent_type(n)) {
                cls.domain |= dm_persistent;
            }
            if (program_helpers::is_or_becomes_output(n)) {
                cls.domain |= dm_user_output;
            }
            bool known = true;
            const size_t bytes = tensor_bytes(t, known);
            if (!known) {
                sizes_known = false;
            } else {
                max_bytes = std::max(max_bytes, bytes);
            }
        }
        if (sizes_known && max_bytes > root_bytes) {
            cls.fallback |= fb_role_conflict;  // a member does not fit the root allocation
        }
        if (cls.root_node != nullptr) {
            if (cls.root.index != 0) {
                cls.fallback |= fb_nonzero_port_root;
            }
            if (program_helpers::is_external_buffer_root(*cls.root_node) && !is_persistent_type(*cls.root_node)) {
                cls.domain |= dm_external_root;
            }
        }

        // readers: users that read a member tensor and are not members themselves
        std::unordered_set<const program_node*> seen;
        for (size_t t : tensors) {
            const size_t ni = tensor_node[t];
            const program_node& n = *nodes[ni];
            const int32_t port = static_cast<int32_t>(t - base[ni]);
            for (const auto* u : n.get_users()) {
                auto ui = idx_of.find(u);
                if (ui == idx_of.end() || member_nodes.count(ui->second)) {
                    continue;
                }
                bool reads = false;
                for (const auto& d : u->get_dependencies()) {
                    if (d.first == &n && std::max(0, d.second) == port) {
                        reads = true;
                        break;
                    }
                }
                if (!reads) {
                    continue;
                }
                if (u->is_type<shape_of>()) {
                    cls.fallback |= fb_shape_of;
                    continue;
                }
                if (seen.insert(u).second) {
                    cls.readers.push_back(u);
                    cls.fallback |= cpu_bits(node_cpu(ui->second));
                }
            }
        }
        std::sort(cls.readers.begin(), cls.readers.end(), [&](const program_node* a, const program_node* b) {
            return res.debug_po_rank(*a) < res.debug_po_rank(*b);
        });

        const size_t class_idx = res.m_classes.size();
        for (const auto& m : cls.members) {
            if (!res.m_class_of_tensor.emplace(key_of(m.id), class_idx).second) {
                ++res.m_duplicate_uids;
                res.m_colliding_keys.insert(key_of(m.id));
            }
            auto& per_port = res.m_class_of_node[m.node];
            if (per_port.size() <= m.id.index) {
                per_port.resize(static_cast<size_t>(m.id.index) + 1, std::numeric_limits<size_t>::max());
            }
            per_port[m.id.index] = class_idx;
        }
        res.m_classes.push_back(std::move(cls));
    }

#ifndef NDEBUG
    // The class partition must contain every owner the settled walk of program_helpers finds (single-port nodes; multi-port
    // consumers of a slot read a specific port and are covered by the per-port edges above).
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nports[i] != 1) {
            continue;
        }
        const auto owner = program_helpers::resolve_buffer_owner(*nodes[i], program_helpers::view_resolution::settled_only);
        auto oi = idx_of.find(owner.owner);
        if (oi == idx_of.end()) {
            continue;
        }
        const auto* a = res.class_of(*nodes[i], 0);
        const auto* b = res.class_of(*owner.owner, owner.owner_port);
        OPENVINO_ASSERT(a != nullptr && a == b, "[GPU] alias_analysis: node ", nodes[i]->id(), " is not in the class of its buffer owner ", owner.owner->id());
    }
#endif
    return res;
}

}  // namespace cldnn

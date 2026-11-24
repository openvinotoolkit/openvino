// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "class_level_memory_dependencies.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "concatenation_inst.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "primitive_inst.h"
#include "reshape_inst.h"
#include "program_helpers.h"
#include "program_node.h"
#include "shape_of_inst.h"

namespace cldnn {

// P1-B-S Debug assertions (zero Release cost). Temporarily define P1B_DEBUG_CHECKS to 1 to exercise them in an NDEBUG build.
#ifndef P1B_DEBUG_CHECKS
#    ifndef NDEBUG
#        define P1B_DEBUG_CHECKS 1
#    endif
#endif

namespace class_level {

namespace {

bool contains_sorted(const std::vector<uint32_t>& v, uint32_t id) {
    return std::binary_search(v.begin(), v.end(), id);
}

}  // namespace

void simple_restriction_store::insert(size_t node, uint32_t id) {
    auto& v = sets[node];
    auto it = std::lower_bound(v.begin(), v.end(), id);
    if (it == v.end() || *it != id) {
        v.insert(it, id);
    }
}

bool simple_restriction_store::contains(size_t node, uint32_t id) const {
    return contains_sorted(sets[node], id);
}

std::string to_string(uint32_t m) {
    static const std::pair<uint32_t, const char*> names[] = {
        {rj_program_level, "program_level"},
        {rj_placement, "placement"},
        {rj_root_kind, "root_kind"},
        {rj_root_not_shareable, "root_not_shareable"},
        {rj_member_kind, "member_kind"},
        {rj_dynamic, "dynamic_or_skippable"},
        {rj_no_mempool, "no_mempool_member"},
        {rj_multi_output, "multi_output"},
        {rj_reused_eltw, "reused_eltw"},
        {rj_ambiguous_uid, "ambiguous_uid"},
        {rj_external_root, "external_root"},
        {rj_shape_of_reader, "shape_of_reader"},
        {rj_duplicate_root_dep, "duplicate_root_dep"},
        {rj_holder_padding, "holder_padding_or_blocked"},
        {rj_mixed_dynamic, "mixed_dynamic"},
    };
    std::string s;
    for (const auto& n : names) {
        if (m & n.first) {
            s += s.empty() ? "" : "|";
            s += n.second;
        }
    }
    return s.empty() ? "admitted" : s;
}

uint32_t pool_admission_reject_mask(const alias_class& k, const program& p, const std::function<bool(uint32_t)>& uid_is_ambiguous) {
    uint32_t m = 0;
    const auto& cfg = p.get_config();
    // L1-0
    if (!cfg.get_enable_concat_holder_pool() || cfg.get_queue_type() != QueueTypes::in_order || p.is_internal_program() || p.is_body_program() ||
        !cfg.get_enable_memory_pool()) {
        m |= rj_program_level;
    }
    // L1-1 (covers CPU / shape_of / role conflicts / persistent / output / external root / non-zero root port)
    if (k.get_placement() != placement::POOL) {
        m |= rj_placement;
    }
    const program_node* root = k.root_node;
    // P1-B-D: a dynamic holder class is only admitted by the sub-switch (the main switch was checked by L1-0)
    const bool dyn_root = root != nullptr && root->is_dynamic();
    const bool dyn_ok = dyn_root && cfg.get_enable_dynamic_concat_holder_pool();
    // L1-2
    if (root == nullptr || !root->is_type<concatenation>() || !root->can_be_optimized() || k.root.index != 0) {
        m |= rj_root_kind;
    }
    // L1-2 (dynamic): the holder is decided per round; only a runtime skippable concat can fall back to a non-fused execution
    if (dyn_ok && !root->is_runtime_skippable()) {
        m |= rj_root_kind;
    }
    // L1-3: the holder must be a poolable node (defect B owner marking and outputs keep it off the pool)
    if (root == nullptr || !root->can_share_buffer() || root->is_output()) {
        m |= rj_root_not_shareable;
    }
    // L1-12: the same producer must not occupy two slots of the holder tree. Leaves are collected through cascaded in-place concats (their
    // dependencies are slots of the holder as well) and through settled views walked backwards (a producer and its own view in two slots).
    if (root != nullptr) {
        std::unordered_set<const program_node*> leaves;
        std::unordered_set<const program_node*> visited_concats;
        std::function<void(const program_node&)> collect = [&](const program_node& concat) {
            if (!visited_concats.insert(&concat).second) {
                m |= rj_duplicate_root_dep;  // the same inner concat reached twice
                return;
            }
            for (const auto& d : concat.get_dependencies()) {
                const program_node* cur = d.first;
                while (!(cur->is_type<concatenation>() && cur->can_be_optimized()) &&
                       program_helpers::is_buffer_view_of_input0(*cur, program_helpers::view_resolution::settled_only)) {
                    cur = cur->get_dependency_with_port(0).first;
                }
                if (cur->is_type<concatenation>() && cur->can_be_optimized()) {
                    collect(*cur);
                } else if (!leaves.insert(cur).second) {
                    m |= rj_duplicate_root_dep;
                }
            }
        };
        collect(*root);
    }
    std::unordered_set<const program_node*> member_nodes;
    for (const auto& mem : k.members) {
        member_nodes.insert(mem.node);
    }
    for (const auto& mem : k.members) {
        const program_node& n = *mem.node;
        // L1-4
        bool kind_ok = (mem.kind == alias_kind::NONE && &n == root) || mem.kind == alias_kind::MUST_SUBREGION || mem.kind == alias_kind::MUST_VIEW;
        if (dyn_ok) {
            // P1-B-D: the envelope degrades the slot edges of a dynamic holder to MAY_SUBREGION and the views of runtime skippable readers to
            // MAY_VIEW (output members are rejected by L1-1 through the placement)
            kind_ok = (mem.kind == alias_kind::NONE && &n == root) ||
                      (mem.kind == alias_kind::MAY_SUBREGION && program_helpers::in_place_concat_holder_of(n) == root) || mem.kind == alias_kind::MAY_VIEW;
        }
        if (!kind_ok) {
            m |= rj_member_kind;
        }
        // L1-5
        if (dyn_ok) {
            if (!n.is_dynamic()) {
                m |= rj_mixed_dynamic;  // runtime skippable members are allowed in a dynamic class
            }
            // L1-13: every byte read from the fused block must be written by a slot this round: no padding, no blocked format
            if ((&n == root || mem.kind == alias_kind::MAY_SUBREGION) && !primitive_inst::holder_layout_is_plain(n.get_output_layout(0))) {
                m |= rj_holder_padding;
            }
        } else if (n.is_dynamic() || n.is_runtime_skippable()) {
            m |= rj_dynamic;
        }
        // L1-6
        if (!n.may_use_mempool()) {
            m |= rj_no_mempool;
        }
        // L1-7
        if (n.get_outputs_count() != 1 || mem.id.index != 0) {
            m |= rj_multi_output;
        }
        // L1-8 (member is a writer)
        if (onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(n) != -1) {
            m |= rj_reused_eltw;
        }
        // L1-9
        if (uid_is_ambiguous(mem.id.uid)) {
            m |= rj_ambiguous_uid;
        }
        // L1-10
        if (program_helpers::is_external_buffer_root(n)) {
            m |= rj_external_root;
        }
    }
    for (const auto* r : k.readers) {
        // L1-5: a dynamic reader may re-decide at runtime (admitted only by the P1-B-D sub-switch in a dynamic class)
        if (dyn_ok) {
            if (!r->is_dynamic()) {
                m |= rj_mixed_dynamic;
            }
        } else if (r->is_dynamic()) {
            m |= rj_dynamic;
        }
        // L1-11
        if (r->is_in_shape_of_subgraph()) {
            m |= rj_shape_of_reader;
        }
        // L1-8: a writer whose reused dependency is a class tensor
        if (const int32_t reused = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(*r); reused != -1) {
            if (member_nodes.count(r->get_dependency_with_port(static_cast<size_t>(reused)).first) != 0) {
                m |= rj_reused_eltw;
            }
        }
    }
    return m;
}

std::vector<std::vector<uint32_t>> snapshot_restrictions(const restriction_store& store) {
    std::vector<std::vector<uint32_t>> res(store.size());
    for (size_t i = 0; i < store.size(); ++i) {
        res[i] = store.values(i);
    }
    return res;
}

void materialize_class_restrictions(restriction_store& store, const std::vector<promoted_class>& classes) {
    if (classes.empty()) {
        return;
    }
    const size_t n = store.size();
    std::unordered_map<uint32_t, std::vector<size_t>> nodes_of_uid;
    for (size_t i = 0; i < n; ++i) {
        nodes_of_uid[store.uid(i)].push_back(i);
    }
    // member uid (the root's own uid included) -> index of the root node
    std::unordered_map<uint32_t, size_t> member_root;
    for (const auto& c : classes) {
        member_root[store.uid(c.root)] = c.root;
        for (size_t m : c.members) {
            member_root[store.uid(m)] = c.root;
        }
    }
    // (2) R(root) |= union of R0(m). Only roots are written, members are read: the order of the classes does not matter.
    for (const auto& c : classes) {
        for (size_t m : c.members) {
            if (m == c.root) {
                continue;
            }
            const std::vector<uint32_t> vals = store.values(m);
            for (uint32_t u : vals) {
                store.insert(c.root, u);
            }
        }
    }
    // (3') every node that is restricted against a member (R1, after all classes finished (2)) is restricted against the class root in
    // both directions. Collected first and applied afterwards, so the scan is not influenced by the insertions.
    std::set<std::pair<size_t, size_t>> pending;  // (node X, root index)
    for (size_t x = 0; x < n; ++x) {
        for (uint32_t u : store.values(x)) {
            auto it = member_root.find(u);
            if (it != member_root.end()) {
                pending.emplace(x, it->second);
            }
        }
    }
    for (const auto& pr : pending) {
        store.insert(pr.first, store.uid(pr.second));
        store.insert(pr.second, store.uid(pr.first));
    }
    // (4) symmetrise: whatever the root is restricted against is restricted against the root, for every node carrying that uid
    for (const auto& c : classes) {
        const uint32_t ruid = store.uid(c.root);
        const std::vector<uint32_t> vals = store.values(c.root);
        for (uint32_t u : vals) {
            auto it = nodes_of_uid.find(u);
            if (it == nodes_of_uid.end()) {
                continue;
            }
            for (size_t j : it->second) {
                store.insert(j, ruid);
            }
        }
    }
}

std::string check_class_invariants(const restriction_store& after, const std::vector<std::vector<uint32_t>>& r0, const std::vector<promoted_class>& classes) {
    std::ostringstream err;
    const size_t n = after.size();
    std::unordered_set<uint32_t> roots;
    std::unordered_map<uint32_t, std::vector<size_t>> by_uid;
    for (const auto& c : classes) {
        roots.insert(after.uid(c.root));
    }
    for (size_t i = 0; i < n; ++i) {
        by_uid[after.uid(i)].push_back(i);
    }
    std::unordered_set<size_t> root_idx;
    for (const auto& c : classes) {
        root_idx.insert(c.root);
    }
    for (const auto& c : classes) {
        const uint32_t ruid = after.uid(c.root);
        // I-B1
        for (size_t m : c.members) {
            const uint32_t muid = after.uid(m);
            for (size_t x = 0; x < n; ++x) {
                const bool related = contains_sorted(r0[m], after.uid(x)) || contains_sorted(r0[x], muid);
                if (!related) {
                    continue;
                }
                if (!contains_sorted(after.values(x), ruid)) {
                    err << "I-B1: node uid " << after.uid(x) << " is related to member uid " << muid << " but root uid " << ruid << " is not in R(X); ";
                }
                if (!contains_sorted(after.values(c.root), after.uid(x))) {
                    err << "I-B1: node uid " << after.uid(x) << " is related to member uid " << muid << " but is not in R(root uid " << ruid << "); ";
                }
            }
        }
        // I-B1-sym
        for (uint32_t u : after.values(c.root)) {
            auto it = by_uid.find(u);
            if (it == by_uid.end()) {
                continue;
            }
            for (size_t j : it->second) {
                if (!contains_sorted(after.values(j), ruid)) {
                    err << "I-B1-sym: uid " << u << " is in R(root uid " << ruid << ") but root uid is not in R(" << u << "); ";
                }
            }
        }
        for (size_t x = 0; x < n; ++x) {
            if (contains_sorted(after.values(x), ruid) && by_uid[after.uid(x)].size() == 1 && !contains_sorted(after.values(c.root), after.uid(x))) {
                err << "I-B1-sym: root uid " << ruid << " is in R(uid " << after.uid(x) << ") but uid " << after.uid(x) << " is not in R(root); ";
            }
        }
    }
    // I-B3: only the root uids may be added to a non-root node; a root only grows
    for (size_t i = 0; i < n; ++i) {
        std::vector<uint32_t> a, b;
        for (uint32_t u : after.values(i)) {
            if (!roots.count(u)) {
                a.push_back(u);
            }
        }
        for (uint32_t u : r0[i]) {
            if (!roots.count(u)) {
                b.push_back(u);
            }
        }
        if (root_idx.count(i)) {
            for (uint32_t u : r0[i]) {
                if (!contains_sorted(after.values(i), u)) {
                    err << "I-B3: root uid " << after.uid(i) << " lost restriction " << u << "; ";
                }
            }
        } else if (a != b) {
            err << "I-B3: non-root uid " << after.uid(i) << " changed beyond root uids; ";
        }
    }
    return err.str();
}

}  // namespace class_level

namespace {

class program_restriction_store : public class_level::restriction_store {
public:
    explicit program_restriction_store(const std::vector<program_node*>& nodes) : m_nodes(nodes) {}
    size_t size() const override {
        return m_nodes.size();
    }
    uint32_t uid(size_t i) const override {
        return static_cast<uint32_t>(m_nodes[i]->get_unique_id());
    }
    const std::vector<uint32_t>& values(size_t i) const override {
        return m_nodes[i]->get_memory_dependencies();
    }
    void insert(size_t i, uint32_t id) override {
        // Deliberately the UNGATED overload: add_memory_dependency(const program_node&) silently skips nodes that do not use the pool.
        if (!m_nodes[i]->has_memory_dependency(id)) {
            m_nodes[i]->add_memory_dependency(std::vector<size_t>{id});
        }
    }

private:
    const std::vector<program_node*>& m_nodes;
};

// zero allocation slot: a plain slot predecessor whose users other than the holder neither view nor may skip its buffer
// (r5-1: alias_analysis builds view edges for input 0 only, a cbo multi-input user may re-bind input_memory(k) at construction).
bool is_zero_alloc_member(const alias_member& m) {
    if (m.kind != alias_kind::MUST_SUBREGION) {
        return false;
    }
    const program_node& n = *m.node;
    // M1: a type whose constructor allocates its own output after the base constructor ran (reshape: `_outputs = allocate_outputs()`).
    // Only reshape does this in src/graph (grep of ctor-level `_outputs =` / `allocate_outputs()`); reorder/permute pass allocate_memory through
    // the base constructor, where the zero-allocation decision applies.
    if (n.is_type<reshape>()) {
        return false;
    }
    const program_node* holder = program_helpers::in_place_concat_holder_of(n);
    for (const auto* u : n.get_users()) {
        if (u == holder) {
            continue;
        }
        if (u->can_be_optimized() || u->is_runtime_skippable()) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace class_level {

std::string check_b4_prime(const program& p) {
    std::ostringstream err;
    auto not_class_route = [&](const program_node& n, const char* what, const program_node& w) {
        const auto r = primitive_inst::allocation_route(n, 0, n.can_be_optimized());
        if (r == primitive_inst::alloc_route::POOL_AS_ROOT || r == primitive_inst::alloc_route::ZERO_ALLOC_MEMBER) {
            err << "B4': " << what << " " << n.id() << " of in-place writer " << w.id() << " has a class route; ";
        }
    };
    for (const auto* w : p.get_processing_order()) {
        const int32_t r = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(*w);
        if (r == -1) {
            continue;
        }
        const program_node& d = w->get_dependency(static_cast<size_t>(r));
        const program_node& o = *program_helpers::resolve_buffer_owner(d, program_helpers::view_resolution::settled_only).owner;
        not_class_route(*w, "writer", *w);
        not_class_route(d, "dependency", *w);
        not_class_route(o, "owner", *w);
    }
    if (const auto* aa = p.get_alias_analysis()) {
        for (const auto& c : aa->classes()) {
            if (c.root_node == nullptr || !c.root_node->is_pool_class_root()) {
                continue;
            }
            if (!c.root_node->can_share_buffer()) {
                err << "B4': promoted root " << c.root_node->id() << " is not shareable; ";
            }
            for (const auto& m : c.members) {
                if (m.kind == alias_kind::MUST_INPLACE || m.kind == alias_kind::MAY_INPLACE) {
                    err << "B4': promoted class of " << c.root_node->id() << " has in-place member " << m.node->id() << "; ";
                }
            }
        }
    }
    return err.str();
}

}  // namespace class_level

void class_level_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::ClassLevelMemoryDependencies");
    const auto& cfg = p.get_config();
    if (!cfg.get_enable_concat_holder_pool()) {
        GPU_DEBUG_IF(cfg.get_enable_dynamic_concat_holder_pool()) {
            GPU_DEBUG_LOG << "[class_level_memory_dependencies] enable_dynamic_concat_holder_pool ignored: enable_concat_holder_pool is off" << std::endl;
        }
        return;
    }
    if (cfg.get_queue_type() != QueueTypes::in_order) {
        GPU_DEBUG_LOG << "[class_level_memory_dependencies] enable_concat_holder_pool ignored: out_of_order queue keeps the LEGACY path" << std::endl;
        return;
    }
    if (p.is_internal_program() || p.is_body_program() || !cfg.get_enable_memory_pool()) {
        return;
    }
    const alias_analysis* aa = p.get_alias_analysis();
    if (aa == nullptr) {
        return;
    }
    [[maybe_unused]] const auto t0 = std::chrono::steady_clock::now();

    std::vector<program_node*> nodes;
    std::unordered_map<const program_node*, size_t> idx_of;
    for (auto* n : p.get_processing_order()) {
        idx_of.emplace(n, nodes.size());
        nodes.push_back(n);
    }

    const auto amb = [&](uint32_t uid) {
        return aa->uid_is_ambiguous(uid);
    };
    std::vector<class_level::promoted_class> promoted;
    std::vector<const alias_class*> promoted_src;
    [[maybe_unused]] size_t rejected = 0;
    for (const auto& c : aa->classes()) {
        if (c.root_node == nullptr || !c.root_node->is_type<concatenation>() || !c.root_node->can_be_optimized()) {
            continue;  // not a holder class: nothing to report either
        }
        const uint32_t mask = class_level::pool_admission_reject_mask(c, p, amb);
        if (mask != 0) {
            ++rejected;
            GPU_DEBUG_LOG << "[class_level_memory_dependencies] class of " << c.root_node->id() << " rejected: " << class_level::to_string(mask) << std::endl;
            continue;
        }
        class_level::promoted_class pc;
        pc.root = idx_of.at(c.root_node);
        for (const auto& m : c.members) {
            pc.members.push_back(idx_of.at(m.node));
        }
        promoted.push_back(std::move(pc));
        promoted_src.push_back(&c);
    }

    if (!promoted.empty()) {
        program_restriction_store store(nodes);
#ifdef P1B_DEBUG_CHECKS
        const auto r0 = class_level::snapshot_restrictions(store);
#endif
        class_level::materialize_class_restrictions(store, promoted);
#ifdef P1B_DEBUG_CHECKS
        const auto err = class_level::check_class_invariants(store, r0, promoted);
        OPENVINO_ASSERT(err.empty(), "[GPU] P1-B-S class restriction invariants violated: ", err);
#endif
        for (size_t i = 0; i < promoted.size(); ++i) {
            program_node& root = *nodes[promoted[i].root];
            const uint32_t ruid = static_cast<uint32_t>(root.get_unique_id());
            root.set_pool_class_root(true);
            root.set_alias_pool_root_uid(ruid);
            for (const auto& m : promoted_src[i]->members) {
                if (m.node != &root && is_zero_alloc_member(m)) {
                    nodes[idx_of.at(m.node)]->set_alias_pool_root_uid(ruid);
                }
            }
        }
#ifdef P1B_DEBUG_CHECKS
        const auto b4 = class_level::check_b4_prime(p);
        OPENVINO_ASSERT(b4.empty(), "[GPU] P1-B-S B4' violated: ", b4);
        for (size_t i = 0; i < promoted.size(); ++i) {
            for (const auto& m : promoted_src[i]->members) {
                const auto route = primitive_inst::allocation_route(*m.node, 0, m.node->can_be_optimized());
                if (m.node == promoted_src[i]->root_node) {
                    OPENVINO_ASSERT(route == primitive_inst::alloc_route::POOL_AS_ROOT, "[GPU] P1-B-S I-B2: root ", m.node->id(), " is not routed as pool root");
                } else if (promoted_src[i]->root_node->is_dynamic()) {
                    // P1-B-D I-B2 (dynamic class): slot predecessors are DIRECT (can_share_buffer(false)); MAY_VIEW members may allocate at runtime
                    // under their own uid (LEGACY_POOL, class restrictions keep their uid and gain the root uid) but never as root / zero-alloc member.
                    // This only covers the compile-time state; the runtime state is covered by the dynamic holder tests.
                    if (m.kind == alias_kind::MAY_SUBREGION) {
                        OPENVINO_ASSERT(route == primitive_inst::alloc_route::DIRECT,
                                        "[GPU] P1-B-D I-B2: slot predecessor ", m.node->id(), " of class ", promoted_src[i]->root_node->id(), " is not DIRECT");
                    }
                    OPENVINO_ASSERT(route != primitive_inst::alloc_route::POOL_AS_ROOT && route != primitive_inst::alloc_route::ZERO_ALLOC_MEMBER,
                                    "[GPU] P1-B-D I-B2: member ", m.node->id(), " of class ", promoted_src[i]->root_node->id(), " has a class route");
                } else {
                    OPENVINO_ASSERT(route != primitive_inst::alloc_route::POOL_AS_ROOT && route != primitive_inst::alloc_route::LEGACY_POOL,
                                    "[GPU] P1-B-S I-B2: member ", m.node->id(), " of class ", promoted_src[i]->root_node->id(), " may call the pool");
                }
            }
        }
#endif
    }
    [[maybe_unused]] const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
    GPU_DEBUG_INFO << "[class_level_memory_dependencies] promoted " << promoted.size() << " class(es), rejected " << rejected << ", " << elapsed_us << " us"
                   << std::endl;
}

}  // namespace cldnn

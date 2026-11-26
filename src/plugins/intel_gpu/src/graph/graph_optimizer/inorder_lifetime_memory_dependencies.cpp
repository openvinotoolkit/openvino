// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a (G+) S4: in-order lifetime restrictions. Design 19 r2, sections 3.3 steps 5 and 6.
//   * make_inorder_leaf_filter(): the predicate of the oooq leaf filter (C1 + C2).
//   * inorder_lifetime_memory_dependencies: the sweep that turns the class intervals of class_lifetimes into restrictions.
// Classes, positions and readers come from alias_analysis / exec_axis through class_lifetimes; nothing here recomputes lifetimes.

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <vector>

#include "alias_analysis.h"
#include "class_lifetime.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "pass_manager.h"
#include "program_node.h"

namespace cldnn {

namespace {

class lifetime_leaf_filter final : public inorder_leaf_filter {
public:
    explicit lifetime_leaf_filter(const class_lifetimes& lt) : m_lt(lt) {}

    bool skip_insert(const program_node& x, const program_node& y) const override {
        if (!eligible(x) || !eligible(y)) {
            return false;
        }
        // C1: nodes that share an alias class (several allocators of one class, e.g. rs siblings plus an output view) keep the oooq entry
        if (!current_inorder_lifetime_mutations().no_same_class_retention && !m_lt.classes_disjoint(x, y)) {
            return false;
        }
        return true;
    }

private:
    bool eligible(const program_node& n) const {
        auto it = m_cache.find(&n);
        if (it == m_cache.end()) {
            it = m_cache.emplace(&n, m_lt.eligible(n)).first;
        }
        return it->second;
    }

    const class_lifetimes& m_lt;
    mutable std::unordered_map<const program_node*, bool> m_cache;
};

}  // namespace

std::unique_ptr<inorder_leaf_filter> make_inorder_leaf_filter(const class_lifetimes& lt) {
    return std::make_unique<lifetime_leaf_filter>(lt);
}

void inorder_lifetime_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::InorderLifetimeMemoryDependencies");
    const auto* aa = p.get_alias_analysis();
    if (aa == nullptr || !m_lt.valid()) {
        return;
    }
    const auto& mut = current_inorder_lifetime_mutations();

    std::unordered_map<const program_node*, program_node*> mutable_of;
    for (auto* n : p.get_processing_order()) {
        mutable_of.emplace(n, n);
    }
    struct positioned {
        uint32_t pos;
        program_node* node;
    };
    std::vector<positioned> nodes;
    for (auto* n : p.get_processing_order()) {
        if (const auto pos = m_lt.position(*n)) {
            nodes.push_back({*pos, n});
        }
    }
    std::sort(nodes.begin(), nodes.end(), [](const positioned& a, const positioned& b) {
        return a.pos < b.pos;
    });

    struct cls {
        const alias_class* k;
        uint64_t birth;
        uint64_t death;   // inclusive; UINT64_MAX for user visible outputs
    };
    std::vector<cls> eligible;
    constexpr uint64_t kInf = std::numeric_limits<uint64_t>::max();
    for (const auto& k : aa->classes()) {
        if (m_lt.class_ineligible_mask(k) != 0 || m_lt.pool_allocators(k).empty()) {
            continue;
        }
        const auto& iv = m_lt.interval_of(k);
        if (!iv.known) {
            continue;
        }
        eligible.push_back({&k, iv.birth, (iv.death_inf && !mut.no_infinity) ? kInf : static_cast<uint64_t>(iv.death)});
    }
    std::sort(eligible.begin(), eligible.end(), [](const cls& a, const cls& b) {
        return a.birth != b.birth ? a.birth < b.birth : a.k < b.k;
    });

    size_t added = 0;
    size_t added_pool = 0;   // statistics: new directed entries between two nodes that really take a pool block (mirrors allocate_output's routing)
    const auto pool_user = [](const program_node& n) {
        return n.may_use_mempool() && n.can_share_buffer() && !n.is_output() && (!n.can_be_optimized() || n.is_runtime_skippable() || n.is_dynamic());
    };
    const auto link = [&](const program_node* a, const program_node* b) {
        if (a == b || a->get_unique_id() == b->get_unique_id()) {
            return;
        }
        program_node* ma = mutable_of.at(a);
        program_node* mb = mutable_of.at(b);
        if (!ma->has_memory_dependency(static_cast<uint32_t>(mb->get_unique_id()))) {
            ma->add_memory_dependency(*mb);
            ++added;
            added_pool += (pool_user(*ma) && pool_user(*mb)) ? 1 : 0;
        }
        if (!mb->has_memory_dependency(static_cast<uint32_t>(ma->get_unique_id()))) {
            mb->add_memory_dependency(*ma);
            ++added;
            added_pool += (pool_user(*ma) && pool_user(*mb)) ? 1 : 0;
        }
    };

    // class x class: closed intervals that intersect (infinity = unbounded)
    std::vector<const cls*> active;
    for (const auto& c : eligible) {
        active.erase(std::remove_if(active.begin(), active.end(),
                                    [&](const cls* a) {
                                        return a->death < c.birth;
                                    }),
                     active.end());
        for (const auto* a : active) {
            for (const auto* x : m_lt.pool_allocators(*a->k)) {
                for (const auto* y : m_lt.pool_allocators(*c.k)) {
                    link(x, y);
                }
            }
        }
        active.push_back(&c);
    }

    // class x node point: a node outside the class that executes inside its interval (its internal buffers live at exactly that position)
    if (!mut.no_point_sweep) {
        for (const auto& c : eligible) {
            auto first = std::lower_bound(nodes.begin(), nodes.end(), c.birth, [](const positioned& a, uint64_t v) {
                return static_cast<uint64_t>(a.pos) < v;
            });
            for (auto it = first; it != nodes.end() && static_cast<uint64_t>(it->pos) <= c.death; ++it) {
                const program_node& n = *it->node;
                if (!n.may_use_mempool() || !m_lt.eligible(n)) {
                    continue;
                }
                bool member = false;
                for (size_t port = 0; port < std::max<size_t>(1, n.get_outputs_count()); ++port) {
                    member = member || aa->class_of(n, static_cast<int32_t>(port)) == c.k;
                }
                if (member) {
                    continue;
                }
                for (const auto* x : m_lt.pool_allocators(*c.k)) {
                    link(x, &n);
                }
            }
        }
    }

    if (m_added != nullptr) {
        *m_added = added;
    }
    GPU_DEBUG_LOG << "[inorder_lifetime_memory_dependencies] eligible classes " << eligible.size() << " of " << aa->classes().size()
                  << ", added_beyond_g " << added << " (between real pool users: " << added_pool << ")" << std::endl;
}

}  // namespace cldnn

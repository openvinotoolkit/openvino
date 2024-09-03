// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include <vector>
#include <list>
#include <algorithm>

using namespace cldnn;

namespace {

class bits_64 {
public:
    explicit bits_64(size_t size, bool set = false) : storage((size / 64) + 1, (set ? ~0ULL : 0ULL)) {}
    bool is_set(size_t idx) const {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        return storage[storage_idx] & mask;
    }
    void set(size_t idx) {
        size_t storage_idx = idx >> 6;
        uint64_t mask = 1ULL << (idx & 0x3F);
        storage[storage_idx] |= mask;
    }
    bool _or(const bits_64& that) {
        bool changed = false;
        size_t sz = std::min(storage.size(), that.storage.size());
        for (size_t i = 0; i < sz; i++) {
            uint64_t myval = storage[i];
            uint64_t thatval = myval | that.storage[i];
            bool local_change = myval != thatval;
            changed |= local_change;
            if (local_change)
                storage[i] = thatval;
        }
        return changed;
    }
#if 0
    void dump(std::ostream& s, size_t cols) {
        size_t idx = 0;
        size_t rows = (storage.size() * 64) / cols;

        s << storage.size() << " items" << std::endl;
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                s << is_set(idx);
                idx++;
            }
            s << std::endl;
        }
    }
#endif

protected:
    std::vector<uint64_t> storage;
};

}  // namespace

void oooq_memory_dependencies::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::OooqMemoryDependencies");
    // For oooq memory dependencies nodes A and B can't share memory if
    // processing_num(A) < processing_num(B) and there is no path from A to B.
    // Assuming precalculation of reachability this function has complexity O(N^2 log N).

    // First create transitive closure of the graph,
    // giving us mapping of node to set of all users that can be reached from this node.
    auto& processing_order = p.get_processing_order();
    std::list<program_node*> processing_order_except_const;
    for (auto n : processing_order) {
        if (!n->is_type<data>()) {
            processing_order_except_const.push_back(n);
        }
    }

    // maps program nodes to bimap vector ids
    auto user_map = std::unordered_map<program_node*, unsigned int>();
    unsigned int processing_order_idx = 0;
    for (auto node : processing_order_except_const) {
        user_map[node] = processing_order_idx++;
    }
    unsigned int num_nodes = static_cast<unsigned int>(user_map.size());

    // full cross ref [node<->node] bitmap.
    // every node has a bit array assigned to it
    // users or the node are marked with 1 bit in this array
    std::vector<bits_64> user_bitmap(num_nodes, bits_64(num_nodes));
    bits_64 suspect_nodes(num_nodes);

    // init bitmaps from direct node users
    for (const auto& node : user_map) {
        for (const auto& user : node.first->get_users()) {
            user_bitmap[node.second].set(user_map.at(user));
        }

        size_t num_dep_nodes = 0;
        for (const auto& dep : node.first->get_dependencies()) {
            if (!dep.first->is_constant()) {
                ++num_dep_nodes;
            }
        }
        if (num_dep_nodes > 1) {
            suspect_nodes.set(user_map.at(node.first));
        }
    }

    // Iteratively extend the users set by adding closure over existing users until no change occurs.
    bool changed = true;
    while (changed) {
        changed = false;
        for (unsigned int n = 0; n < num_nodes; n++) {
            auto& users = user_bitmap[n];

            for (unsigned int user_id = n + 1; user_id < num_nodes; user_id++) {
                if (users.is_set(user_id)) {
                    changed |= users._or(user_bitmap[user_id]);
                }
            }
        }
    }

    // Connection query:
    auto are_connected = [&](unsigned int A, unsigned int B) {
        return user_bitmap[A].is_set(B);
    };

    unsigned int A = 0;
    auto itr_A = processing_order_except_const.begin();

    while (itr_A != processing_order_except_const.end()) {
        if (suspect_nodes.is_set(A)) {
            std::vector<std::pair<program_node*, unsigned int>> deps;
            for (const auto& dep : (*itr_A)->get_dependencies()) {
                if (!dep.first->is_type<data>()) {
                    deps.emplace_back(dep.first, user_map.at(dep.first));
                }
            }

            std::sort(deps.begin(), deps.end(),
                    [](const std::pair<cldnn::program_node*, unsigned int>& a, const std::pair<cldnn::program_node*, unsigned int>& b) {
                        return a.second < b.second;
                    });

            for (size_t i = 0; i < deps.size(); ++i) {
                for (size_t j = i + 1; j < deps.size(); ++j) {
                    if (are_connected(deps[i].second, deps[j].second)) {
                        for (const auto& user : deps[j].first->get_users()) {
                            add_memory_dependency(deps[i].first, user);
                            add_memory_dependency(user, deps[i].first);
                        }
                    }
                }
            }
        }
        unsigned int B = ++A;
        auto itr_B = ++itr_A;
        while (itr_B != processing_order_except_const.end()) {
            if (!are_connected(A, B)) {
                add_memory_dependency(*itr_A, *itr_B);
                add_memory_dependency(*itr_B, *itr_A);
            } else {
                for (auto u : (*itr_A)->get_users()) {
                    if (u != (*itr_B) && !are_connected(B, user_map[u]) && !are_connected(user_map[u], B)) {
                        add_memory_dependency(*itr_A, *itr_B);
                        add_memory_dependency(*itr_B, *itr_A);
                        break;
                    }
                }
            }
            itr_B++;
            B++;
        }
    }
}

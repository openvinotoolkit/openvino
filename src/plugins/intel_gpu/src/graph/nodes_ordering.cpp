// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "program_node.h"

namespace cldnn {
// helper method for calc_processing order
void program::nodes_ordering::calc_processing_order_visit(program_node* node) {
    if (node->is_marked()) {
        return;
    }
    for (auto* user : node->users) {
        calc_processing_order_visit(user);
    }
    node->mark();
    _processing_order.push_front(node);
    processing_order_iterators[node] = _processing_order.begin();
}

// DFS to sort nodes topologically
// any topological sort of nodes is required for further optimizations
void program::nodes_ordering::calc_processing_order(program& p) {
    _processing_order.clear();
    for (auto* input : p.get_inputs()) {
        calc_processing_order_visit(input);
    }
    for (auto& node : _processing_order) {
        node->unmark();
    }
}

void program::nodes_ordering::calculate_in_order_processing_order(program& p) {
    const auto previous_order = _processing_order;
    std::map<program_node*, int32_t> distances;
    for (auto* node : previous_order) {
        int32_t distance = 0;
        for (const auto& dep : node->get_dependencies()) {
            distance = std::max(distance, distances[dep.first] + 1);
        }
        distances[node] = distance;
    }

    clear();
    std::unordered_set<program_node*> visited;
    std::function<void(program_node*)> visit = [&](program_node* node) {
        if (!visited.insert(node).second)
            return;

        auto dependencies = node->get_dependencies();
        std::stable_sort(dependencies.begin(), dependencies.end(), [&](const auto& lhs, const auto& rhs) {
            return distances[lhs.first] > distances[rhs.first];
        });
        for (const auto& dep : dependencies) {
            visit(dep.first);
        }

        _processing_order.push_back(node);
        processing_order_iterators[node] = std::prev(_processing_order.end());
    };

    for (auto* output : p.get_outputs()) {
        visit(output);
    }
    for (auto* node : previous_order) {
        visit(node);
    }
}

/*
    recalculate processing_order
    algorithm based on: CLRS 24.5 (critical path in DAG)
    modifications: adjust for multiple inputs
    input: any topological order in processing order
    output: BFS topological order.
    */
void program::nodes_ordering::calculate_BFS_processing_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("calculate_BFS_processing_order");
    std::map<program_node*, int> distances;
    for (auto* itr : _processing_order) {
        distances[itr] = -1;
    }
    int max_distance = 0;
    for (auto* itr : _processing_order) {
        // Init
        if (distances[itr] == -1) {  // this must be an input
            distances[itr] = 0;      // initialize input
        }
        // RELAX
        for (const auto& user : itr->get_users()) {
            distances[user] = std::max(distances[user], distances[itr] + 1);
            max_distance = std::max(max_distance, distances[user]);
        }
    }

    // bucket sort nodes based on their max distance from input
    std::vector<std::vector<program_node*>> dist_lists;
    dist_lists.resize(max_distance + 1);
    for (auto* itr : _processing_order) {
        dist_lists[distances[itr]].push_back(itr);
    }

    // replace the old processing order by the new one, still topological.
    _processing_order.clear();
    for (auto& dist : dist_lists) {
        for (auto& node : dist) {
            _processing_order.push_back(node);
            processing_order_iterators[node] = _processing_order.end();
            processing_order_iterators[node]--;
        }
    }
}

// verifies if a given node will be processed before all its dependent nodes
bool program::nodes_ordering::is_correct(program_node* node) {
    for (const auto& dep : node->get_dependencies()) {
        if (get_processing_number(node) < get_processing_number(dep.first)) {
            return false;
        }
    }
    return true;
}

void program::nodes_ordering::save(cldnn::BinaryOutputBuffer& ob) const {
    ob << _processing_order.size();
    auto itr = rbegin();
    while (itr != rend()) {
        const auto& node = *itr;
        ob << node->id();
        itr++;
    }
}

void program::nodes_ordering::load(cldnn::BinaryInputBuffer& ib, program& p) {
    size_t num_nodes;
    ib >> num_nodes;

    clear();
    for (size_t i = 0; i < num_nodes; ++i) {
        primitive_id node_id;
        ib >> node_id;

        auto* node = p.get_node_ptr(node_id).get();
        _processing_order.push_front(node);
        processing_order_iterators[node] = _processing_order.begin();
    }
}
}  // namespace cldnn

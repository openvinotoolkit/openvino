/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "program_impl.h"
#include "program_node.h"
#include "error_handler.h"
#include <vector>
#include <map>
#include <algorithm>

namespace cldnn {
// helper method for calc_processing order
void program_impl::nodes_ordering::calc_processing_order_visit(program_node* node) {
    if (node->is_marked())
        return;
    for (auto user : node->users) {
        calc_processing_order_visit(user);
    }
    node->mark();
    _processing_order.push_front(node);
    processing_order_iterators[node] = _processing_order.begin();
    return;
}

// DFS to sort nodes topologically
// any topological sort of nodes is required for further optimizations
void program_impl::nodes_ordering::calc_processing_order(program_impl& p) {
    _processing_order.clear();
    for (auto input : p.get_inputs()) {
        calc_processing_order_visit(input);
    }
    for (auto& node : _processing_order) {
        node->unmark();
    }
    return;
}

/*
    recalculate processing_order
    algorithm based on: CLRS 24.5 (critical path in DAG)
    modifications: adjust for multiple inputs
    input: any topological order in processing order
    output: BFS topological order.
    */
void program_impl::nodes_ordering::calculate_BFS_processing_order() {
    std::map<program_node*, int> distances;
    for (auto itr : _processing_order) {
        distances[itr] = -1;
    }
    int max_distance = 0;
    for (auto itr : _processing_order) {
        // Init
        if (distances[itr] == -1) {  // this must be an input
            distances[itr] = 0;      // initialize input
        }
        // RELAX
        for (auto& user : itr->get_users()) {
            distances[user] = std::max(distances[user], distances[itr] + 1);
            max_distance = std::max(max_distance, distances[user]);
        }
    }

    // bucket sort nodes based on their max distance from input
    std::vector<std::vector<program_node*>> dist_lists;
    dist_lists.resize(max_distance + 1);
    for (auto itr : _processing_order) {
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
    return;
}

// verifies if a given node will be processed before all its dependent nodes
bool program_impl::nodes_ordering::is_correct(program_node* node) {
    for (auto& dep : node->get_dependencies()) {
        if (get_processing_number(node) < get_processing_number(dep)) {
            return false;
        }
    }
    return true;
}
}  // namespace cldnn

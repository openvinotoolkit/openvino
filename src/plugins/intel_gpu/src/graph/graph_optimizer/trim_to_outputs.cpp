// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"

// ToDo: remove those include with the appropriate code below once we will have support for multiple outputs of a
// primitive
#include "pooling_inst.h"
#include <vector>
#include <queue>

using namespace cldnn;

// This pass optimizes out nodes which have no impact on outputs
void trim_to_outputs::run(program& p) {
    const size_t actual_nodes = p.get_processing_order().size();
    if (actual_nodes == 0 || actual_nodes == p.get_outputs().size()) {
        return;
    }

    // do backward bfs starting from all outputs
    std::queue<std::vector<program_node*>> queue;
    queue.push(p.get_outputs());

    std::vector<program_node*> special_nodes;
    for (auto& node : p.get_processing_order()) {   // input layout may become disconnected during prior boxes calculations so
        if (node->is_type<input_layout>()) {        // it may have not been marked at this place but we don't want to remove it
            special_nodes.push_back(node);          // ToDo: remove this after support for multi-outputs in primitives will
        }                                           // be implemented.
    }
    queue.push(special_nodes);

    while (!queue.empty()) {
        auto nodes_list = queue.front();
        queue.pop();

        for (auto& node : nodes_list) {
            if (!node->is_marked()) {
                node->mark();
                if (!node->get_dependencies().empty()) {
                   std::vector<program_node*> deps;
                    for (auto& dep : node->get_dependencies()) {
                        deps.push_back(dep.first);
                    }
                    queue.push(deps);
                }
            }
        }
    }

    // all not-marked nodes should be removed
    std::vector<program_node*> to_rem;
    for (auto& node : p.get_processing_order()) {
        if (!node->is_marked())
            to_rem.push_back(node);
    }
    p.remove_nodes(to_rem);

    for (auto& node : p.get_processing_order()) {
        node->unmark();
    }
}

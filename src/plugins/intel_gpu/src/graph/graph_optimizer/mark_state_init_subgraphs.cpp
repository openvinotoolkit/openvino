// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "read_value_inst.h"
#include "pass_manager.h"
#include <queue>

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_state_init_subgraphs::mark_init_subgraph(program& p, read_value_node& node) {
    const auto& variable_id = node.get_primitive()->variable_id;
    if (p.contains_state(variable_id))
        return;

    std::queue<program_node*> q;
    q.push(&node);

    auto can_be_marked = [&](const program_node* dep_node) {
        if (p.has_state_initializers(variable_id, dep_node->id()))
            return false;

        for (auto& u : dep_node->get_users()) {
            if (u == &node)
                continue;
            if (p.has_state_initializers(variable_id, u->id()))
                continue;
            else
                return false;
        }
        GPU_DEBUG_TRACE_DETAIL << "marked " << dep_node->id() << " as node in a init_subgraph for " << node.id() << std::endl;
        return true;
    };

    while (!q.empty()) {
        auto cur_size = q.size();
        for (size_t i = 0; i < cur_size; ++i) {
            auto& cur_node = q.front();
            q.pop();
            for (auto& dep : cur_node->get_dependencies()) {
                if (can_be_marked(dep.first)) {
                    p.set_state_initializers(variable_id, dep.first->id());
                    q.push(dep.first);
                }
            }
        }
    }
}

void mark_state_init_subgraphs::run(program& p) {
    auto rit = p.get_processing_order().rbegin();
    for (; rit != p.get_processing_order().rend(); rit++) {
        auto& node = *rit;
        if (node->is_type<read_value>()) {
            mark_init_subgraph(p, node->as<read_value>());
        }
    }
}

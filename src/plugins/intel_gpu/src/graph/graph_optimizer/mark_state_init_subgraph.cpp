// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "read_value_inst.h"
#include "pass_manager.h"
#include <queue>

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_state_init_subgraphs::mark_node(program_node* node) {
    if (node->is_in_state_init_subgraph())
        return;
    if (!node->is_type<read_value>())
        return;

    node->set_state_init_subgraph(node);

    std::queue<program_node*> Q;
    Q.push(node);
    auto can_be_marked = [&](const program_node* dep_node, const program_node* cur_node) {
        for (auto& u : dep_node->get_users()) {
            if (u == cur_node)
                continue;
            if (u->get_state_of_init_subgraph() != node) {
                return false;
            }
        }
        GPU_DEBUG_TRACE_DETAIL << "marked" << dep_node->id() << " as node in a init_subgraph for " << node->id()
                               << std::endl;
        return true;
    };

    while (!Q.empty()) {
        auto cur_size = Q.size();
        for (size_t i = 0; i < cur_size; ++i) {
            auto& cur_node = Q.front();
            Q.pop();
            for (auto& dep : cur_node->get_dependencies()) {
                if (can_be_marked(dep.first, cur_node)) {
                    dep.first->set_state_init_subgraph(node);
                    Q.push(dep.first);
                }
            }
        }
    }
}

void mark_state_init_subgraphs::run(program& p) {
    auto rit = p.get_processing_order().rbegin();
    if (p.is_new_shape_infer()) {
        for (; rit != p.get_processing_order().rend(); rit++) {
            auto node = *rit;
            mark_node(node);
        }
    }
}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"

// ToDo: remove those include with the appropriate code below once we will have support for multiple outputs of a
// primitive
#include "max_unpooling_inst.h"
#include "pooling_inst.h"
#include <vector>
#include <queue>

using namespace cldnn;

// This pass optimizes out nodes which have no impact on outputs
void trim_to_outputs::run(program_impl& p) {
    const size_t actual_nodes = p.get_processing_order().size();
    if (actual_nodes == 0 || actual_nodes == p.get_outputs().size()) {
        return;
    }

    // do backward bfs starting from all outputs
    std::queue<const std::vector<program_node*>*> queue;
    queue.push(&p.get_outputs());

    std::vector<program_node*> special_nodes;
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<input_layout>() ||  // input layout may become disconnected during prior boxes calculations so
                                              // it may have not been marked at this place but we don't want to remove it
            node->is_type<max_unpooling>() ||  // ToDo: remove this after support for multi-outputs in primitives will
                                               // be implemented.
            (node->is_type<pooling>() && node->as<pooling>().get_primitive()->mode == pooling_mode::max_with_argmax))
            special_nodes.push_back(node);
    }
    queue.push(&special_nodes);

    while (!queue.empty()) {
        auto nodes_list = queue.front();
        queue.pop();

        for (auto& node : *nodes_list) {
            if (!node->is_marked()) {
                node->mark();
                if (!node->get_dependencies().empty()) {
                    queue.push(&node->get_dependencies());
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
}
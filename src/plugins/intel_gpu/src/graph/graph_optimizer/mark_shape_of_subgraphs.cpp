// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "reshape_inst.h"
#include "broadcast_inst.h"
#include "tile_inst.h"
#include "pass_manager.h"

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_shape_of_subgraphs::look_for_shape_of_subgraph(program_node& node, program_node& parent_shape_of) {
    bool shape_of_node = node.is_type<shape_of>();

    if (shape_of_node)
        mark_node(node, parent_shape_of);

    // Check if all dependencies are constant or marked as a part of shape_of subgraphs
    bool can_execute_in_subgraph = true;
    for (auto& dependency : node.get_dependencies()) {
        if (!dependency.first->is_in_shape_of_subgraph() && !dependency.first->is_constant()) {
            can_execute_in_subgraph = false;
            break;
        }
    }

    // Check if current node is a shape infer dependency of any of node's users
    bool is_shape_infer_dep = node.is_shape_infer_dep();

    if (!can_execute_in_subgraph && !is_shape_infer_dep && !shape_of_node)
        return;

    if (!can_mark_node(node))
        return;

    mark_node(node, parent_shape_of);

    for (auto& user : node.get_users())
        look_for_shape_of_subgraph(*user, parent_shape_of);
}

bool mark_shape_of_subgraphs::can_mark_node(program_node& node) {
    if (node.has_fused_primitives())
        return false;

    if (node.is_type<reshape>())
        return true;

    impl_types prev_impl = node.get_preferred_impl_type();

    node.set_preferred_impl_type(impl_types::cpu);
    bool cpu_impl_found = (!node.is_dynamic() && node.type()->does_possible_implementation_exist(node)) ||
                          (node.is_dynamic() && node.type()->does_dynamic_implementation_exist(node));

    node.set_preferred_impl_type(prev_impl);

    if (cpu_impl_found)
        return true;

    return false;
}

void mark_shape_of_subgraphs::mark_node(program_node& node, program_node& parent_shape_of) {
    node.set_in_shape_of_subgraph(true);
    node.add_dependant_shape_of_node(&parent_shape_of);
    for (auto dep : node.get_dependencies()) {
        if (dep.first->is_in_shape_of_subgraph()) {
            for (auto shape_of : dep.first->get_dependant_shape_of_nodes()) {
                node.add_dependant_shape_of_node(shape_of);
            }
        }
    }

    const auto default_subgraph_impl = impl_types::cpu;
    if (_update_impls)
        if (!node.is_type<reshape>())
            node.set_preferred_impl_type(default_subgraph_impl);
}

void mark_shape_of_subgraphs::run(program& p) {
    if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        for (auto& node : p.get_processing_order()) {
            if (node->is_type<shape_of>())
                look_for_shape_of_subgraph(*node, *node);
        }
    }
}

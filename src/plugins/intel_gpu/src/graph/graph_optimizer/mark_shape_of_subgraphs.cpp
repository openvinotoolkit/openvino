// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_inst.h"
#include "reshape_inst.h"
#include "eltwise_inst.h"
#include "pass_manager.h"

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_shape_of_subgraphs::look_for_shape_of_subgraph(program_node& node) {
    if (node.is_type<shape_of>()) {
        mark_node(node);
        return;
    }

    // Check if all dependencies are constant or marked as a part of shape_of subgraph
    bool can_execute_in_subgraph = true;
    bool has_shape_of_subgraph_dep = false;
    for (auto& dependency : node.get_dependencies()) {
        if (dependency.first->is_in_shape_of_subgraph()) {
            has_shape_of_subgraph_dep = true;
        } else if (!dependency.first->is_constant()) {
            can_execute_in_subgraph = false;
            break;
        }
    }

    // Node should have at least one dependency marked as a part of shape_of subgraph
    if (!has_shape_of_subgraph_dep || !can_execute_in_subgraph)
        return;

    if (!can_mark_node(node))
        return;

    mark_node(node);
}

bool mark_shape_of_subgraphs::can_mark_node(const program_node& node) {
    if (node.has_fused_primitives())
        return false;

    if (node.is_type<reshape>())
        return true;

    // Exclude eltwise with boolean mode types since CPU reference implementation
    // couldn't save result in int8 data type (as it requested by GPU plugin,
    // because we use it instead of boolean data type)
    if (node.is_type<eltwise>()) {
        auto& eltwise_node = node.as<eltwise>();
        auto eltwise_mode = eltwise_node.get_primitive()->mode;
        if (eltwise::eltwise_bool_modes.find(eltwise_mode) != eltwise::eltwise_bool_modes.end())
            return false;
    }

    auto available_impls = node.type()->get_available_impls(node);
    auto cpu_impl_found = available_impls.find(impl_types::cpu) != available_impls.end();

    if (cpu_impl_found)
        return true;

    return false;
}

void mark_shape_of_subgraphs::mark_node(program_node& node) {
    node.set_in_shape_of_subgraph(true);

    // If current node has shape_of type add it to dependant shape_of nodes for
    // correct dependency propagation for users
    if (node.is_type<shape_of>())
        node.add_dependant_shape_of_node(&node);

    // Add parent shape_of nodes from other dependencies if there are any
    for (auto dep : node.get_dependencies()) {
        if (dep.first->is_in_shape_of_subgraph()) {
            for (auto shape_of : dep.first->get_dependant_shape_of_nodes()) {
                node.add_dependant_shape_of_node(shape_of);
            }
        }
    }

    // Update impl if needed
    const auto default_subgraph_impl = impl_types::cpu;
    if (_update_impls)
        if (!node.is_type<reshape>())
            node.set_preferred_impl_type(default_subgraph_impl);
}

void mark_shape_of_subgraphs::run(program& p) {
    if (p.get_config().get_property(ov::intel_gpu::allow_new_shape_infer)) {
        for (auto& node : p.get_processing_order()) {
            look_for_shape_of_subgraph(*node);
        }
    }
}

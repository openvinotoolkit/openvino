// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_inst.h"
#include "shape_of_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "eltwise_inst.h"
#include "select_inst.h"
#include "strided_slice_inst.h"
#include "gather_inst.h"
#include "input_layout_inst.h"
#include "paged_attention_inst.h"
#include "pass_manager.h"

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

static bool is_shape_of_subgraph_root(program_node& node) {
    if (node.is_type<shape_of>()) {
        return true;
    }

    // Allow input_layout to be the root of the shape_of subgraph if it's 'max_context_len'
    // input of PagedAttention, which can be used as a shape calculation flow source in some
    // models like Qwen and Qwen2
    if (node.is_type<input_layout>()) {
        const auto& users = node.get_users();
        for (const auto& user : users) {
            const auto max_context_len_input_id = 12;
            if (user->is_type<paged_attention>() && user->get_dependency_index(node) == max_context_len_input_id) {
                return true;
            }
        }
    }

    return false;
}

void mark_shape_of_subgraphs::look_for_shape_of_subgraph(program_node& node) {
    if (is_shape_of_subgraph_root(node)) {
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

    // read_value may have initializer which is shape_of sub-graph, but read_value itself is not a part of such sub-graph
    if (node.is_type<read_value>())
        return false;

    // CPU implementation does not support float data types for mask and mixed types for data inputs, so check them
    // before including it into shape_of sub-graph
    if (node.is_type<select>() &&
        (data_type_traits::is_floating_point(node.get_input_layout(0).data_type) ||
         node.get_input_layout(1).data_type != node.get_input_layout(2).data_type))
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

    // Exclude gather_compressed primitive because gather_cpu_impl doesn't support it.
    if (node.is_type<gather>()) {
        auto& gather_node = node.as<gather>();
        auto gather_compressed_weight_mode = gather_node.get_primitive()->compressed_weights;
        if (gather_compressed_weight_mode)
            return false;
    }

    // Exclude stride_slice primitive if it's input is big const ternsor, else CPU reference implementation
    // will lead to huge performance drop.
    if (node.is_type<strided_slice>() && node.get_dependency(0).is_constant() &&
        node.get_dependency(0).get_output_layout().count() > 128 * 1024) {
        return false;
    }

    // skip mark_node for broadcast node if dependency nodes are data and shape_of
    auto& dependencies = node.get_dependencies();
    if (node.is_type<broadcast>() && dependencies.size() == 2) {
        if (dependencies[0].first->is_type<data>() && dependencies[1].first->is_type<shape_of>() && (dependencies[1].first->get_users().size() == 1))
            return false;
    }

    return true;
}

void mark_shape_of_subgraphs::mark_node(program_node& node) {
    node.set_in_shape_of_subgraph(true);

    // If current node has shape_of type add it to dependant shape_of nodes for
    // correct dependency propagation for users
    if (is_shape_of_subgraph_root(node))
        node.add_dependant_shape_of_node(&node);

    // Add parent shape_of nodes from other dependencies if there are any
    for (auto dep : node.get_dependencies()) {
        if (dep.first->is_in_shape_of_subgraph()) {
            for (auto shape_of : dep.first->get_dependant_shape_of_nodes()) {
                node.add_dependant_shape_of_node(shape_of);
            }
        }
    }
}

void mark_shape_of_subgraphs::run(program& p) {
    if (p.is_new_shape_infer()) {
        for (auto& node : p.get_processing_order()) {
            look_for_shape_of_subgraph(*node);
        }
    }
}

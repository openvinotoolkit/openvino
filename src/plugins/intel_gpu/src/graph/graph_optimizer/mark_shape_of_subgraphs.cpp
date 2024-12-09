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
#include "pass_manager.h"

#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

namespace {
    bool has_input_layout_dep(const std::vector<std::pair<cldnn::program_node*, int>>& shape_of_deps) {
        for (auto& shape_of_dep : shape_of_deps) {
            // input_layout node
            if (shape_of_dep.first->is_type<input_layout>()) {
                return true;
            }
        }
        return false;
    }

    bool has_shape_of_dep(const std::vector<std::pair<cldnn::program_node*, int>>& broadcast_deps) {
        for (auto& broadcast_dep : broadcast_deps) {
            // shape_of node
            if (broadcast_dep.first->is_type<shape_of>()) {
                auto& shape_of_deps = broadcast_dep.first->get_dependencies();
                return has_input_layout_dep(shape_of_deps);
            }
        }
        return false;
    }

    bool has_broadcast_dep(const std::vector<std::pair<cldnn::program_node*, int>>& reorder_deps) {
        for (auto& reorder_dep : reorder_deps) {
            // broadcast node
            if (reorder_dep.first->is_type<broadcast>()) {
                auto& broadcast_deps = reorder_dep.first->get_dependencies();
                return has_shape_of_dep(broadcast_deps);
            }
        }
        return false;
    }

    bool has_reorder_reoder_dep(const std::vector<std::pair<cldnn::program_node*, int>>& eltwise_deps) {
        for (auto& eltwise_dep : eltwise_deps) {
            // reorder node (reorder -> eltwise)
            if (eltwise_dep.first->is_type<reorder>()) {
                auto& eltwise_dep_reorder_deps = eltwise_dep.first->get_dependencies();

                for (auto& eltwise_dep_reorder_dep : eltwise_dep_reorder_deps) {
                    // reorder node (broadcast -> reorder)
                    if (eltwise_dep_reorder_dep.first->is_type<reorder>()) {
                        auto& reorder_dep_reorder_deps = eltwise_dep_reorder_dep.first->get_dependencies();
                        return has_broadcast_dep(reorder_dep_reorder_deps);
                    }
                }
            }
        }
        return false;
    }

    bool has_eltwise_dep(const std::vector<std::pair<cldnn::program_node*, int>>& reorder_deps) {
        for (auto& reorder_dep : reorder_deps) {
            // eltwise node
            if (reorder_dep.first->is_type<eltwise>()) {
                auto& eltwise_deps = reorder_dep.first->get_dependencies();
                return has_reorder_reoder_dep(eltwise_deps);
            }
        }
        return false;
    }

    bool has_reorder_dep(const std::vector<std::pair<cldnn::program_node*, int>>& conv_deps) {
        for (auto& conv_dep : conv_deps) {
            //if (conv_dep.first->id().find(dequantize_name) != std::string::npos) {

            // reorder node ( reorder -> convolution)
            if (conv_dep.first->is_type<reorder>()) {
                auto& reorder_deps = conv_dep.first->get_dependencies();
                return has_eltwise_dep(reorder_deps);
            }
        }
        return false;
    }

    bool has_convolution_dep(const std::vector<std::pair<cldnn::program_node*, int>>& dependencies) {
        for (auto& dependency : dependencies) {
            // convolution node
            if (dependency.first->is_type<convolution>()) {
                auto& conv_deps = dependency.first->get_dependencies();
                return has_reorder_dep(conv_deps);
            }
        }
        return false;
    }

    // check dependencies for reorder node added for convolution in quantized model
    bool skip_quantization_conv_reorder(const program_node& node) {
        // reorder -> convolution -> reorder -> eltwise -> reorder -> reorder -> broadcast -> shape_of -> input_layout
        if (!node.is_type<reorder>()) {
            return false;
        }

        auto& dependencies = node.get_dependencies();
        return has_convolution_dep(dependencies);
    }

} // namespace

void mark_shape_of_subgraphs::look_for_shape_of_subgraph(program_node& node) {
    if (node.is_type<shape_of>()) {
        mark_node(node);
        return;
    }

    // skip mark_node for reorder node (after convolution node) for quantized model
    if (skip_quantization_conv_reorder(node)) {
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

    return true;
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
}

void mark_shape_of_subgraphs::run(program& p) {
    if (p.is_new_shape_infer()) {
        for (auto& node : p.get_processing_order()) {
            look_for_shape_of_subgraph(*node);
        }
    }
}

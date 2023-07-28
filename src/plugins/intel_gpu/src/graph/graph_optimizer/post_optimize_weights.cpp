// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "implementation_map.hpp"

#include "convolution_inst.h"
#include "binary_convolution_inst.h"
#include "deconvolution_inst.h"
#include "deformable_convolution_inst.h"
#include "fully_connected_inst.h"
#include "lstm_dynamic_input_inst.h"

namespace cldnn {

post_optimize_weights::post_optimize_weights(reorder_factory& rf_ref)
    : base_pass("post_optimize_weights"), _rf(rf_ref) {}

template<typename T> post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset(const T& node) {
    return weights_bias_offset(node.get_primitive()->input.size(), program_helpers::wrap_if_single(node.get_primitive()->weights).size());
}

template <>
post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset<lstm_dynamic_input_node>(const lstm_dynamic_input_node& node) {
    return weights_bias_offset(node.get_primitive()->input.size() + 1, program_helpers::wrap_if_single(node.get_primitive()->weights).size());
}

// function which prepares given primitive for weights optimization
template<typename T>
void post_optimize_weights::optimize_weights(T& node, program& p) {
    auto offsets = get_weights_bias_offset(node);
    auto impl = node.get_selected_impl();

    // Skip load-time weights reordering if impl is not selected
    if (!impl)
        return;

    if (impl->is_dynamic())
        return;

    // Don't run impl selection to avoid double compilation of reorder kernels
    // in main program and internal program for constant propagation
    auto set_implementation = [&p, &impl](program_node& weights_reorder_node) {
        if (!weights_reorder_node.is_constant()) {
            auto factory = WeightsReordersFactory::get(impl_types::ocl, shape_types::static_shape);
            auto reorder_kernel_params = impl->get_weights_reorder_kernel_params();
            reorder_kernel_params->prog = &p;
            auto reorder_impl = factory(*reorder_kernel_params);

            weights_reorder_node.set_selected_impl(reorder_impl->clone());
            if (auto impl = weights_reorder_node.get_selected_impl()) {
                auto params = weights_reorder_node.get_kernel_impl_params();
                p.get_kernels_cache().add_kernels_source(*params, impl->get_kernels_source());
            }
        }
    };

    auto output_layout = node.get_output_layout();
    auto weights_reorder_params = impl->get_weights_reorder_params();
    for (auto i = offsets.weights_offset; i < offsets.bias_offset; i++) {
        program_node& prev_node = node.get_dependency(i);

        if (weights_reorder_params != nullptr) {
            bool can_be_fused = prev_node.is_type<reorder>() &&
                                prev_node.get_users().size() == 1 &&
                                prev_node.get_dependencies().size() == 1 &&
                                !prev_node.has_fused_primitives() &&
                                !prev_node.as<reorder>().has_mean() &&
                                prev_node.as<reorder>().get_primitive()->subtract_per_feature.empty();
            if (can_be_fused) {
                // Need to update input data_type for correct merging format reorder with precision reorder
                data_types input_dtype = prev_node.get_input_layouts()[0].data_type;
                auto updated_input_layout = weights_reorder_params->get_input_layout();
                updated_input_layout.data_type = input_dtype;
                weights_reorder_params->set_input_layout(updated_input_layout);

                auto weights_reorder = _rf.get_weights_reorder(prev_node.get_primitive()->input[0].pid,
                                                               weights_reorder_params);
                auto& weights_reorder_node = p.get_or_create(weights_reorder.first);
                p.replace(prev_node, weights_reorder_node);
                weights_reorder_node.recalc_output_layout(false);

                if (!weights_reorder.second) {
                    set_implementation(weights_reorder_node);
                }
            } else {
                auto weights_reorder = _rf.get_weights_reorder(prev_node.id(), weights_reorder_params);
                // insert new weights reorder node to topology
                p.add_intermediate(weights_reorder.first, node, i, !weights_reorder.second);
                // set weights reorder's node output layout and implementation
                auto& weights_reorder_node = node.get_dependency(i);
                weights_reorder_node.get_output_layout(false);

                if (!weights_reorder.second) {
                    set_implementation(weights_reorder_node);
                }
            }
        }
    }
    // set the old output layout and do not invalidate users as change of weights will not affect output layout
    node.set_output_layout(output_layout, false);
}

void post_optimize_weights::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<convolution>()) {
            optimize_weights(node->as<convolution>(), p);
        } else if (node->is_type<binary_convolution>()) {
            optimize_weights(node->as<binary_convolution>(), p);
        } else if (node->is_type<deconvolution>()) {
            optimize_weights(node->as<deconvolution>(), p);
        } else if (node->is_type<deformable_conv>()) {
            optimize_weights(node->as<deformable_conv>(), p);
        } else if (node->is_type<fully_connected>()) {
            optimize_weights(node->as<fully_connected>(), p);
        } else if (node->is_type<lstm_dynamic_input>()) {
            optimize_weights(node->as<lstm_dynamic_input>(), p);
        }
    }
}

}  // namespace cldnn

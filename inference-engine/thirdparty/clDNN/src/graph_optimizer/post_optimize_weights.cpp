// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "program_helpers.h"
#include "cldnn/primitives/fused_conv_eltwise.hpp"
#include "include/fused_conv_eltwise_inst.h"
#include "include/binary_convolution_inst.h"
#include "include/deformable_convolution_inst.h"
#include "lstm_dynamic_input_inst.h"

namespace cldnn {

post_optimize_weights::post_optimize_weights(reorder_factory& rf_ref)
    : base_pass("post_optimize_weights"), _rf(rf_ref) {}

// function which prepares given primitive for weights optimization
template<typename T> post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset(const T& node) {
    return weights_bias_offset(node.get_primitive()->input.size(), program_helpers::wrap_if_single(node.get_primitive()->weights).size());
}

template <>
post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset<fused_conv_eltwise_node>(const fused_conv_eltwise_node& node) {
    return weights_bias_offset(node.get_primitive()->input.size(), program_helpers::wrap_if_single(node.get_primitive()->conv.weights).size());
}

template <>
post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset<lstm_dynamic_input_node>(const lstm_dynamic_input_node& node) {
    return weights_bias_offset(node.get_primitive()->input.size() + 1, program_helpers::wrap_if_single(node.get_primitive()->weights).size());
}

// function which prepares given primitive for weights optimization
template<typename T>
void post_optimize_weights::optimize_weights(T& node, program_impl& p) {
    auto offsets = get_weights_bias_offset(node);
    auto impl = node.get_selected_impl();
    auto output_layout = node.get_output_layout();
    auto& weights_reorder_params = impl->_weights_reorder_params;

    for (auto i = offsets.weights_offset; i < offsets.bias_offset; i++) {
        auto& weights_node = node.get_dependency(i);
        auto weights_layout = weights_node.get_output_layout();

        auto reorders = _rf.get_weights_reorder(weights_node.id(), weights_layout, weights_reorder_params);

        for (auto& reorder : reorders) {
            // insert new generic_layer node to topology
            p.add_intermediate(reorder.first, node, i, !reorder.second);
            // set generic_layer's node output layout and implementation
            auto& g_node = node.get_dependency(i);
            g_node.get_output_layout(false);

            // Don't run impl selection to avoid double compilation of reorder kernels
            // in main program and internal program for constant propagation
            if (!g_node.is_constant())
                g_node.selected_impl = g_node.type()->choose_impl(g_node);
        }
    }

    // Reset weights reorder params to not keep source code pointer
    weights_reorder_params.engine = kernel_selector::generic_kernel_params::Engine::NONE;
    weights_reorder_params.clKernel = nullptr;
    weights_reorder_params.cpuKernel = nullptr;

    // set the old output layout and do not invalidate users as change of weights will not affect output layout
    node.set_output_layout(output_layout, false);
}

void post_optimize_weights::run(program_impl& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->type() == convolution::type_id()) {
            optimize_weights(node->as<convolution>(), p);
        }
        if (node->type() == binary_convolution::type_id()) {
            optimize_weights(node->as<binary_convolution>(), p);
        } else if (node->type() == deconvolution::type_id()) {
            optimize_weights(node->as<deconvolution>(), p);
        } else if (node->type() == deformable_conv::type_id()) {
            optimize_weights(node->as<deformable_conv>(), p);
        } else if (node->type() == fully_connected::type_id()) {
            optimize_weights(node->as<fully_connected>(), p);
        } else if (node->type() == fused_conv_eltwise::type_id()) {
            optimize_weights(node->as<fused_conv_eltwise>(), p);
        } else if (node->type() == lstm_dynamic_input::type_id()) {
            optimize_weights(node->as<lstm_dynamic_input>(), p);
        }
    }
}

}  // namespace cldnn

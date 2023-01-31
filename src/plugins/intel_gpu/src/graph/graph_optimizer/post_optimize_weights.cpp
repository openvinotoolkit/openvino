// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "convolution_inst.h"
#include "binary_convolution_inst.h"
#include "deconvolution_inst.h"
#include "deformable_convolution_inst.h"
#include "fully_connected_inst.h"
#include "lstm_dynamic_input_inst.h"

namespace cldnn {

post_optimize_weights::post_optimize_weights(reorder_factory& rf_ref)
    : base_pass("post_optimize_weights"), _rf(rf_ref) {}

// function which prepares given primitive for weights optimization
void post_optimize_weights::optimize_weights(program_node& node, program& p, size_t weights_idx) {
    auto impl = node.get_selected_impl();

    // Skip load-time weights reordering if impl is not selected
    if (!impl)
        return;

    if (impl->is_dynamic())
        return;

    auto output_layout = node.get_output_layout();
    auto weights_reorder_params = impl->get_weights_reorder_params();

    auto& weights_node = node.get_dependency(weights_idx);
    auto weights_layout = weights_node.get_output_layout();

    auto reorder = _rf.get_weights_reorder(weights_node.id(), weights_layout, weights_reorder_params);

    // insert new generic_layer node to topology
    p.add_intermediate(reorder.first, node, weights_idx, !reorder.second);
    // set generic_layer's node output layout and implementation
    auto& g_node = node.get_dependency(weights_idx);
    g_node.get_output_layout(false);

    // Don't run impl selection to avoid double compilation of reorder kernels
    // in main program and internal program for constant propagation
    if ((!g_node.is_constant()) && (!reorder.second)) {
        g_node.set_selected_impl(g_node.type()->choose_impl(g_node));
        if (auto impl = g_node.get_selected_impl()) {
            auto params = g_node.get_kernel_impl_params();
            p.get_kernels_cache().add_kernels_source(*params, impl->get_kernels_source());
        }
    }

    // Reset weights reorder params to not keep source code pointer
    impl->reset_weights_reorder_params();

    // set the old output layout and do not invalidate users as change of weights will not affect output layout
    node.set_output_layout(output_layout, false);
}

void post_optimize_weights::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<convolution>() ||
            node->is_type<binary_convolution>() ||
            node->is_type<deconvolution>() ||
            node->is_type<deformable_conv>() ||
            node->is_type<fully_connected>()) {
            optimize_weights(*node, p, 1);
        } else if (node->is_type<lstm_dynamic_input>()) {
            optimize_weights(*node, p, 2);
        }
    }
}

}  // namespace cldnn

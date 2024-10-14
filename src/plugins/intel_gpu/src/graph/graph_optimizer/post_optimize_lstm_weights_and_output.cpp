// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "impls/registry/implementation_map.hpp"

#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/runtime/format.hpp"
#include "lstm_seq_inst.h"
#include "intel_gpu/primitives/mutable_data.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif // ENABLE_ONEDNN_FOR_GPU

namespace cldnn {

post_optimize_lstm_weights_and_output::post_optimize_lstm_weights_and_output(reorder_factory& rf_ref)
    : base_pass("post_optimize_lstm_weights_and_output"), _rf(rf_ref) {}

template<typename T> post_optimize_lstm_weights_and_output::weights_bias_offset post_optimize_lstm_weights_and_output::get_weights_bias_offset(const T& node) {
    return weights_bias_offset(3, 6);
}

// function which prepares given primitive for weights optimization
template<typename T>
void post_optimize_lstm_weights_and_output::optimize_lstm_weights(T& node, program& p) {
    //auto offsets = get_weights_bias_offset(node);
    auto impl = node.get_selected_impl();

    // Skip load-time weights reordering if impl is not selected
    if (!impl)
        return;

    if (impl->is_dynamic()) {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->disable_build_time_weight_reorder_for_dynamic_nodes) {
            return;
        }
        // TODO: To relax current limitation w.r.t the future optimization of weight reorder process
        // In dynamic shape, selected weight format can change in runtime. However reordering blocked format to blocked format is not fully verified yet.
        // So we need to enable other primitives such as convolution with verifying reorder b/w the possible layouts
        // Also we skip weight reorder for onednn impl because onednn fully connected layer is using simple format, therefore
        // reordering to cldnn shape_agnostic_kernel's preferred blocked format at build time does not helpful for the performance.
        // This situation might be changed once onednn shape agnostic kernel is used in the future.
        if (p.is_internal_program())
            return;
        if (node.get_preferred_impl_type() == impl_types::onednn)
            return;
        if (node.type() != fully_connected::type_id())
            return;
    }

    auto output_layout = node.get_output_layout();
    auto weights_reorder_params = impl->get_weights_reorder_params();
    for (auto i = 3; i < 6; i++) {
        program_node& prev_node = node.get_dependency(i);
        if (weights_reorder_params != nullptr) {
                if (i != 5) {
                    _rf.get_weights_split(prev_node.id(), weights_reorder_params, p, prev_node, node, i);
                } else {
                    _rf.get_bias_split(prev_node.id(), weights_reorder_params, p, prev_node, node);
                }
                // insert new weights reorder node to topology
                //p.add_intermediate(weights_reorder.first, node, i, !weights_reorder.second);
                // set weights reorder's node output layout and implementation
                auto& weights_reorder_node = node.get_dependency(i);
                weights_reorder_node.get_output_layout(false);
        }
    }
    // set the old output layout and do not invalidate users as change of weights will not affect output layout
    node.set_output_layout(output_layout, false);
}

void post_optimize_lstm_weights_and_output::run(program& p) {
    for (auto& node : p.get_processing_order()) {
        if (node->is_type<lstm_seq>()) {
            optimize_lstm_weights(node->as<lstm_seq>(), p);
        }
    }
    p.get_processing_order().calc_processing_order(p);
    int i = 0;
    for (auto node : p.get_processing_order()) {
        if (node->is_type<cldnn::mutable_data>()) {
            continue;
        }
        for (auto prev_node : node->get_dependencies()) {
            if (prev_node.first->is_type<lstm_seq>()) {
                auto impl = prev_node.first->get_selected_impl();
                 if (!impl)
                    continue;
                auto weights_reorder_params = impl->get_weights_reorder_params();
                if (weights_reorder_params == nullptr) {
                    continue;
                }
                prev_node.first->recalc_output_layouts(false);
                _rf.get_out_reorder(p, prev_node.first, node, i);
                node->recalc_output_layouts(false);
                i++;
            }
        }
    }
    p.get_processing_order().calc_processing_order(p);
}

}  // namespace cldnn

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_helpers.h"
#include "impls/registry/registry.hpp"

#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/runtime/format.hpp"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif // ENABLE_ONEDNN_FOR_GPU
namespace cldnn {

post_optimize_weights::post_optimize_weights(reorder_factory& rf_ref)
    : base_pass("post_optimize_weights"), _rf(rf_ref) {}

template<typename T> post_optimize_weights::weights_bias_offset post_optimize_weights::get_weights_bias_offset(const T& node) {
    return weights_bias_offset(node.get_primitive()->input.size(), program_helpers::wrap_if_single(node.get_primitive()->weights).size());
}

// function which prepares given primitive for weights optimization
template<typename T>
void post_optimize_weights::optimize_weights(T& node, program& p) {
    auto offsets = get_weights_bias_offset(node);
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
    // Don't run impl selection to avoid double compilation of reorder kernels
    // in main program and internal program for constant propagation
    auto set_implementation = [&p, &impl](program_node& weights_reorder_node) {
        if (!weights_reorder_node.is_constant()) {
            auto reorder_kernel_params = impl->get_weights_reorder_kernel_params();
            weights_reorder_node.set_preferred_impl_type(impl_types::any);
            auto reorder_impl = weights_reorder_node.type()->create_impl(weights_reorder_node);

            weights_reorder_node.set_selected_impl(std::move(reorder_impl));
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
                                prev_node.as<reorder>().is_simple_reorder() &&
                                prev_node.get_users().size() == 1 &&
                                prev_node.get_dependencies().size() == 1 &&
                                (format::is_weights_format(prev_node.get_input_layout().format) ||
                                 format::is_simple_data_format(prev_node.get_input_layout().format));

            if (can_be_fused) {
                // Need to update input data_type for correct merging format reorder with precision reorder
                auto updated_input_layout = weights_reorder_params->get_input_layout();
                data_types input_dtype = prev_node.get_input_layout().data_type;
                updated_input_layout.data_type = input_dtype;

                // Need to update input format in case of fusing weights constant with transpose
                format input_fmt = prev_node.get_input_layout().format;
                updated_input_layout.format = from_weights_layout(to_weights_layout(input_fmt, false));

                weights_reorder_params->set_input_layout(updated_input_layout);
#ifdef ENABLE_ONEDNN_FOR_GPU
                // Need to update WeightsReorderParamsOneDNN of fc onednn imple when input layout data_type/format is different
                auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(weights_reorder_params);
                if (onednn_weights_params &&
                   (updated_input_layout.format != onednn::find_data_format(onednn_weights_params->_in_desc) ||
                    onednn::convert_data_type(updated_input_layout.data_type) != onednn_weights_params->_in_desc.get_data_type())) {
                    onednn_weights_params->_in_desc = onednn::layout_to_memory_desc(updated_input_layout);
                }
#endif // ENABLE_ONEDNN_FOR_GPU
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
        } else if (node->is_type<deconvolution>()) {
            optimize_weights(node->as<deconvolution>(), p);
        } else if (node->is_type<fully_connected>()) {
            optimize_weights(node->as<fully_connected>(), p);
        }
    }
}

}  // namespace cldnn
